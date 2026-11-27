use super::*;
use polylogue_model::{
    Author, Reference, RevisionReference, Timestamp,
    annotation::{
        AnnotationKind, AnnotationPatch, Change, LinkRole, NewAnnotation, NewAnnotationLink,
    },
    authorship::{Authorship, Channel},
    user::{UserRequest, UserWrite},
};
use polylogue_store::{ReadPool, verify};
use polylogue_testkit::{ClaudeHome, claude_assistant, claude_user};
use std::sync::Arc;

fn read_sources(home: &ClaudeHome, sources: bool) {
    let (mut engine, _lock) = home.engine(1 << 20);
    engine.config.reads_sources = sources;
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    engine.writer().flush().unwrap();
    assert!(engine.archive.index_build().is_none());
}

fn catch_up(home: &ClaudeHome, from: u64, to: u64) -> u64 {
    let (mut engine, _lock) = home.engine(1 << 20);
    engine.config.reads_sources = false;
    let pass = engine
        .catch_up_archive_suffix(Position(from), Position(to))
        .unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    engine.writer().flush().unwrap();
    pass.messages_written
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &to.join(entry.file_name()));
        } else {
            fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
}

fn fork(home: &mut ClaudeHome, name: &str) -> PathBuf {
    let later = home.dir.path().join(name);
    copy_tree(&home.root, &later);
    std::mem::replace(&mut home.root, later)
}

/// The converter changes the identity of each copy independently. Manufacture
/// that input here after opening and advancing the real archive through the
/// engine so the cuts have the same sequence and byte layout.
fn prepare_later(home: &ClaudeHome, id: &str) {
    let conn = Connection::open(home.root.join("archive.db")).unwrap();
    meta_set(&conn, "archive_id", id).unwrap();
    drop(conn);
    polylogue_testkit::delete_index(&home.root);
}

fn original_bytes(root: &Path) -> Vec<u8> {
    let archive = Archive::at(root);
    let pool = ReadPool::open(&archive).unwrap();
    let mut lines = Vec::new();
    let result = verify::verify(
        &pool,
        &Arc::new(archive.blobs().unwrap()),
        Some(&mut lines),
        &mut |work| {
            work();
            Ok(())
        },
    )
    .unwrap();
    assert!(result.problems.is_empty(), "{result:?}");
    lines
}

fn add_vector(root: &Path, text_hash: u8) {
    let conn = Connection::open(root.join("embeddings.db")).unwrap();
    conn.execute(
        "INSERT INTO embedded(text_hash, model, dims, vector, embedded_at)
         VALUES (?1,'example/synthetic/2/document/unit',2,x'0000803f00000000',1)",
        [vec![text_hash; 32]],
    )
    .unwrap();
}

fn author(home: &ClaudeHome, id: &str, write: UserWrite) -> String {
    let (engine, _lock) = home.engine(1 << 20);
    let writer = engine.writer();
    let result = writer
        .archive()
        .user(UserRequest {
            id: id.into(),
            machine: "test-machine".into(),
            at: Timestamp::from_unix_ms(1).unwrap(),
            authorship: Authorship {
                author: Author::Operator,
                via: Channel::Cli,
            },
            write,
        })
        .unwrap();
    writer.flush().unwrap();
    result.id
}

fn search_hits(root: &Path, word: &str) -> i64 {
    let index = open_read(&root.join("index.db")).unwrap();
    (0..4)
        .map(|i| {
            index
                .query_row(
                    &format!("SELECT count(*) FROM search_{i}.text_fts WHERE text_fts MATCH ?1"),
                    [word],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap()
        })
        .sum()
}

#[test]
fn reuses_a_checked_index_and_reads_the_append_from_retained_bytes() {
    let mut home = ClaudeHome::new();
    home.append("kept.jsonl", &claude_user("one", "kept", "prior horizon"));
    read_sources(&home, true);
    add_vector(&home.root, 1);
    let note = author(
        &home,
        "first-note",
        UserWrite::Assert {
            annotations: vec![NewAnnotation {
                kind: AnnotationKind::Note,
                key: None,
                value: Some("first note".into()),
                valid_until: None,
                links: vec![NewAnnotationLink {
                    role: LinkRole::About,
                    reference: Reference::parse("session:claude-code:kept").unwrap(),
                }],
            }],
        },
    );
    assert_eq!(home.texts("kept"), ["prior horizon"]);
    let earlier = fork(&mut home, "later");

    home.append(
        "kept.jsonl",
        &claude_assistant("two", "kept", "later marsh"),
    );
    home.append("new.jsonl", &claude_user("three", "new", "new input"));
    read_sources(&home, true);
    add_vector(&home.root, 2);
    let revised = author(
        &home,
        "revised-note",
        UserWrite::ReviseAnnotations {
            bases: vec![RevisionReference::exact(note.clone(), note.clone())],
            patch: AnnotationPatch {
                value: Change::Set("revised note".into()),
                ..Default::default()
            },
        },
    );
    let manifest = original_bytes(&home.root);
    assert_eq!(std::str::from_utf8(&manifest).unwrap().lines().count(), 3);
    prepare_later(&home, "1234567890abcdef1234567890abcdef");
    let durable = fs::read(home.root.join("archive.db")).unwrap();
    let paid = fs::read(home.root.join("embeddings.db")).unwrap();

    let (from, to) = reuse(&earlier, &home.root).unwrap();
    let expected_id = ArchiveId("1234567890abcdef1234567890abcdef".into());
    assert_eq!(
        IndexFiles::read(&home.root).unwrap().installed,
        Installed::Current {
            archive: expected_id.clone(),
            position: Position(from),
        }
    );
    assert_eq!(fs::read(home.root.join("archive.db")).unwrap(), durable);
    assert_eq!(fs::read(home.root.join("embeddings.db")).unwrap(), paid);
    assert_eq!(search_hits(&home.root, "horizon"), 1);
    assert_eq!(search_hits(&home.root, "marsh"), 0);

    for name in ["kept.jsonl", "new.jsonl"] {
        fs::rename(home.file(name), home.dir.path().join(name)).unwrap();
    }
    catch_up(&home, from, to);
    assert_eq!(home.texts("kept"), ["prior horizon", "later marsh"]);
    assert_eq!(home.texts("new"), ["new input"]);
    assert_eq!(search_hits(&home.root, "horizon"), 1);
    assert_eq!(search_hits(&home.root, "marsh"), 1);
    let annotation = RevisionReference {
        root: note,
        revision: None,
    };
    let state = home
        .read(|reader| reader.annotation_family_state(&annotation))
        .unwrap();
    assert_eq!(
        state.heads,
        [polylogue_model::Reference::Annotation(
            RevisionReference::exact(annotation.root, revised)
        )]
    );
    assert_eq!(original_bytes(&home.root), manifest);
    assert_eq!(
        IndexFiles::read(&home.root).unwrap().installed,
        Installed::Current {
            archive: expected_id.clone(),
            position: Position(to),
        }
    );

    // Once caught up, the same checked index can seed another later cut.
    let intermediate = fork(&mut home, "final");
    for name in ["kept.jsonl", "new.jsonl"] {
        fs::rename(home.dir.path().join(name), home.file(name)).unwrap();
    }
    home.append(
        "suffix.jsonl",
        &claude_user("four", "suffix", "final suffix"),
    );
    read_sources(&home, true);
    prepare_later(&home, "abcdef1234567890abcdef1234567890");
    let (from, final_position) = reuse(&intermediate, &home.root).unwrap();
    assert_eq!(from, to);
    assert!(final_position > from);
    fs::remove_file(home.file("suffix.jsonl")).unwrap();
    catch_up(&home, from, final_position);
    assert_eq!(catch_up(&home, from, final_position), 0);
    assert_eq!(home.texts("suffix"), ["final suffix"]);
}

#[test]
fn refuses_different_catalog_rows_or_segment_bytes_before_installing() {
    let mut home = ClaudeHome::new();
    home.append("kept.jsonl", &claude_user("one", "kept", "prior horizon"));
    read_sources(&home, true);
    let earlier = fork(&mut home, "later");
    home.append(
        "kept.jsonl",
        &claude_assistant("two", "kept", "later marsh"),
    );
    read_sources(&home, true);
    prepare_later(&home, "1234567890abcdef1234567890abcdef");

    let conn = Connection::open(home.root.join("archive.db")).unwrap();
    conn.execute("UPDATE revision SET seen_at=0 WHERE number=1", [])
        .unwrap();
    assert!(
        reuse(&earlier, &home.root)
            .unwrap_err()
            .to_string()
            .contains("revision")
    );
    assert!(!home.index_db().exists());
    conn.execute("UPDATE revision SET seen_at=(SELECT seen_at FROM observation WHERE revision=1) WHERE number=1", []).unwrap();
    drop(conn);

    let segment = home.root.join("segments/current.hb");
    let mut bytes = fs::read(&segment).unwrap();
    bytes[0] ^= 1;
    fs::write(segment, bytes).unwrap();
    assert!(
        reuse(&earlier, &home.root)
            .unwrap_err()
            .to_string()
            .contains("differs at or after byte")
    );
    assert!(!home.index_db().exists());
}

#[test]
fn continues_one_native_source_after_preparing_the_checked_cut() {
    let mut home = ClaudeHome::new();
    home.append("prior.jsonl", &claude_user("one", "prior", "checked cedar"));
    read_sources(&home, true);
    add_vector(&home.root, 1);
    let earlier = fork(&mut home, "source");

    home.append("warm.jsonl", &claude_user("two", "warm", "warm harbor"));
    read_sources(&home, true);
    let candidate = home.dir.path().join("candidate");
    // Use the same protocol on hosts whose test scratch supports reflinks.
    let probe = home.dir.path().join("probe.db");
    let strategy = if reflink_database(&home.root.join("archive.db"), &probe).is_ok() {
        reflink_database
    } else {
        copy_database
    };
    let pending = {
        let _running = lock(&home.root).unwrap();
        continuation::Pending::prepare_using(&earlier, &home.root, &candidate, strategy).unwrap()
    };
    assert!(pending.checked_at > pending.index_at);
    assert!(!candidate.exists());

    home.append("final.jsonl", &claude_user("three", "final", "final reef"));
    read_sources(&home, true);
    let manifest = original_bytes(&home.root);
    let (_, record_position) = open_archive(&Archive::at(&home.root)).unwrap();
    let archive = Archive::at(&home.root);
    let source_vectors = polylogue_store::vectors::Vectors::writer(&archive, 1024).unwrap();
    {
        let _running = lock(&home.root).unwrap();
        source_vectors
            .put_query_vector(
                "example/paid",
                "final reef",
                &[1., 2.],
                &polylogue_model::WallClock::new(jiff::Timestamp::now),
            )
            .unwrap();
    }
    // Keep the committed vector WAL present across the capture.
    assert!(home.root.join("embeddings.db-wal").is_file());
    let (from, to) = pending.finish().unwrap();
    assert_eq!(to, record_position);
    let marker = "segments/current.hb.committed";
    assert_eq!(
        fs::read(candidate.join(marker)).unwrap(),
        fs::read(home.root.join(marker)).unwrap()
    );
    assert_eq!(from, open_archive(&Archive::at(&earlier)).unwrap().1);
    assert!(to > from);
    drop(source_vectors);
    let archive = Archive::at(&candidate);
    let vectors = polylogue_store::vectors::Vectors::snapshot(&archive, 1024).unwrap();
    assert_eq!(
        vectors.query_vector("example/paid", "final reef").unwrap(),
        Some(vec![1., 2.])
    );
    assert_eq!(
        open_read(&archive.embeddings_db())
            .unwrap()
            .query_row("SELECT count(*) FROM embedded", [], |row| row
                .get::<_, u64>(0))
            .unwrap(),
        1
    );
    drop(vectors);

    for name in ["prior.jsonl", "warm.jsonl", "final.jsonl"] {
        fs::remove_file(home.file(name)).unwrap();
    }
    home.root = candidate;
    assert_eq!(search_hits(&home.root, "cedar"), 1);
    assert_eq!(search_hits(&home.root, "reef"), 0);
    catch_up(&home, from, to);
    assert_eq!(home.texts("warm"), ["warm harbor"]);
    assert_eq!(home.texts("final"), ["final reef"]);
    assert_eq!(search_hits(&home.root, "reef"), 1);
    assert_eq!(original_bytes(&home.root), manifest);
    assert_eq!(
        IndexFiles::read(&home.root).unwrap().installed,
        Installed::Current {
            archive: archive_id(&open_archive(&archive).unwrap().0).unwrap(),
            position: Position(to)
        }
    );
}

#[test]
fn continuation_refuses_a_replaced_branch_or_regressed_record() {
    let mut home = ClaudeHome::new();
    home.append("prior.jsonl", &claude_user("one", "prior", "checked cedar"));
    read_sources(&home, true);
    let earlier = fork(&mut home, "source");
    let candidate = home.dir.path().join("candidate");
    let prepare = || {
        continuation::Pending::prepare_using(&earlier, &home.root, &candidate, copy_database)
            .unwrap()
    };
    let pending = prepare();
    {
        let _running = lock(&home.root).unwrap();
        assert!(
            pending
                .finish()
                .unwrap_err()
                .to_string()
                .contains("source writer lock")
        );
        assert!(!candidate.exists());
    }
    let pending = prepare();

    let segment = home.root.join("segments/current.hb");
    let old_segment = home.root.join("segments/old.hb");
    fs::rename(&segment, &old_segment).unwrap();
    fs::copy(&old_segment, &segment).unwrap();
    assert!(
        pending
            .finish()
            .unwrap_err()
            .to_string()
            .contains("was replaced")
    );
    assert!(!candidate.exists());

    let pending = prepare();
    let (conn, position) = open_archive(&Archive::at(&home.root)).unwrap();
    drop(conn);
    let conn = Connection::open(home.root.join("archive.db")).unwrap();
    meta_set(&conn, "write_counter", (position - 1).to_string()).unwrap();
    drop(conn);
    assert!(
        pending
            .finish()
            .unwrap_err()
            .to_string()
            .contains("record position regressed")
    );
    assert!(!candidate.exists());
}
