use super::*;

fn missing(vectors: &Vectors, key: &str, hash: &[u8], chunks: usize) -> HashSet<u32> {
    vectors
        .missing_chunks(key, &[(hash.to_vec(), chunks)])
        .unwrap()
        .remove(hash)
        .unwrap()
}

/// Index text rows for `hashes`.
fn texts(archive: &Archive, hashes: &[&[u8]]) {
    let index =
        crate::archive::open_write(&archive.index_db(), crate::archive::cache::SETUP).unwrap();
    for hash in hashes {
        index
            .execute("INSERT INTO text(hash, body) VALUES (?1, x'00')", [hash])
            .unwrap();
    }
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

#[test]
fn stores_scans_and_survives_reopening() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
    texts(&archive, &[&[1], &[2], &[3]]);
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut v = reader.vectors_writer().unwrap();
    let key = "local/test/3/document/unit";
    v.put(
        key,
        &[
            Vector {
                text_hash: vec![1],
                chunk: 0,
                values: unit(&[1.0, 0.0, 0.0]),
            },
            Vector {
                text_hash: vec![2],
                chunk: 0,
                values: unit(&[0.0, 1.0, 0.0]),
            },
            // A second chunk of text 2, nearer the query: the text's
            // best chunk counts.
            Vector {
                text_hash: vec![2],
                chunk: 1,
                values: unit(&[0.9, 0.1, 0.0]),
            },
        ],
    )
    .unwrap();
    v.fail(key, &[3], "too long").unwrap();
    drop(v);
    drop(reader);
    drop(pool);
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let v = reader.vectors().unwrap();
    let loaded = std::sync::Arc::new(v.load(key).unwrap());
    assert_eq!(loaded.vectors(), 3);
    assert_eq!(loaded.revision, v.revision(key).unwrap());
    let near = loaded.similarity(&unit(&[1.0, 0.05, 0.0])).unwrap();
    let (one, two) = (near.of(&[1]).unwrap(), near.of(&[2]).unwrap());
    assert!(one > two, "{one} {two}");
    // Text 2's nearer second chunk counts, not its first.
    assert!(two > 0.9, "{two}");
    assert!(near.of(&[3]).is_none());
    assert!(missing(&v, key, &[1], 1).is_empty());
    assert!(missing(&v, key, &[2], 2).is_empty());
    assert_eq!(v.rejected(key).unwrap(), HashSet::from([vec![3]]));
    let s = v.status().unwrap();
    assert_eq!(s.keys[0].texts, 2);
    assert_eq!(s.keys[0].vectors, 3);
    assert_eq!(s.keys[0].failures, 1);
    // Another key's vectors never answer this one.
    let other = std::sync::Arc::new(v.load("other").unwrap());
    assert_eq!(other.vectors(), 0);
    assert!(
        other
            .similarity(&[1.0, 0.0, 0.0])
            .unwrap()
            .of(&[1])
            .is_none()
    );
}

#[test]
fn readers_share_one_budget_with_their_single_optional_vector_connection() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
    let budget = (8 << 20) + 123;
    let pool = crate::ReadPool::with_budget(&archive, 3, budget).unwrap();
    let readers: Vec<_> = (0..3).map(|_| pool.reader().unwrap()).collect();
    let mut total = 0;
    for reader in &readers {
        for schema in ["main", "a"] {
            let row = reader
                .query(&format!("PRAGMA {schema}.cache_size"), &[])
                .unwrap();
            total += row[0]["cache_size"].as_i64().unwrap().unsigned_abs() * 1024;
        }
        let vectors = reader.vectors().unwrap();
        let cache: i64 = vectors
            .conn
            .pragma_query_value(None, "cache_size", |r| r.get(0))
            .unwrap();
        let mmap: i64 = vectors
            .conn
            .pragma_query_value(None, "mmap_size", |r| r.get(0))
            .unwrap();
        assert!(cache < 0);
        assert_eq!(mmap, 0);
        total += cache.unsigned_abs() * 1024;
        assert!(reader.vectors().is_err());
        assert!(reader.vectors_writer().is_err());
    }
    assert!(total <= budget);
    assert!(budget - total < 3 * 3 * 1024);
}

#[test]
fn vector_write_upgrade_keeps_its_cache_and_its_read_results() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
    texts(&archive, &[&[1]]);
    let pool = crate::ReadPool::with_budget(&archive, 1, 3 << 20).unwrap();
    let reader = pool.reader().unwrap();
    assert!(reader.vectors().unwrap().status().unwrap().keys.is_empty());
    let mut vectors = reader.vectors_writer().unwrap();
    let cache: i64 = vectors
        .conn
        .pragma_query_value(None, "cache_size", |r| r.get(0))
        .unwrap();
    let mmap: i64 = vectors
        .conn
        .pragma_query_value(None, "mmap_size", |r| r.get(0))
        .unwrap();
    assert_eq!(cache, -1024);
    assert_eq!(mmap, 0);
    vectors
        .put(
            "synthetic",
            &[Vector {
                text_hash: vec![1],
                chunk: 0,
                values: vec![1.0],
            }],
        )
        .unwrap();
    drop(vectors);
    assert_eq!(
        reader.vectors().unwrap().status().unwrap().keys[0].vectors,
        1
    );
}

#[test]
fn the_smallest_vector_writer_budget_keeps_its_total_cache_target() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
    texts(&archive, &[&[1; 32], &[2; 32]]);
    let pool = crate::ReadPool::with_budget(&archive, 1, 3 * 1024).unwrap();
    let reader = pool.reader().unwrap();
    let mut vectors = reader.vectors_writer().unwrap();
    let main: i64 = vectors
        .conn
        .pragma_query_value(None, "cache_size", |r| r.get(0))
        .unwrap();
    assert_eq!(main, -1);
    vectors
        .put(
            "synthetic",
            &[Vector {
                text_hash: vec![1; 32],
                chunk: 0,
                values: vec![1.0],
            }],
        )
        .unwrap();
    vectors
        .fail("synthetic", &[2; 32], "synthetic failure")
        .unwrap();
    assert!(missing(&vectors, "synthetic", &[1; 32], 1).is_empty());
    assert!(
        vectors
            .rejected("synthetic")
            .unwrap()
            .contains(&vec![2; 32])
    );
}

#[test]
fn pl_similarity_answers_only_inside_its_query() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
    texts(&archive, &[&[1], &[2]]);
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let key = "local/test/2/document/unit";
    reader
        .vectors_writer()
        .unwrap()
        .put(
            key,
            &[Vector {
                text_hash: vec![1],
                chunk: 0,
                values: unit(&[1.0, 0.0]),
            }],
        )
        .unwrap();
    let loaded = std::sync::Arc::new(reader.vectors().unwrap().load(key).unwrap());
    let similarity = std::sync::Arc::new(loaded.similarity(&[1.0, 0.0]).unwrap());
    let sql = "SELECT hex(hash) AS h, pl_similarity(hash) AS s FROM text ORDER BY hash";
    let rows = reader.with_similarity(similarity, || reader.query(sql, &[]).unwrap());
    assert!(
        (rows[0]["s"].as_f64().unwrap() - 1.0).abs() < 0.01,
        "{rows:?}"
    );
    assert!(rows[1]["s"].is_null());
    let outside = reader.query(sql, &[]).unwrap();
    assert!(outside[0]["s"].is_null());
}

#[test]
fn purchased_vectors_are_kept_when_their_text_has_no_current_index_row() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut vectors = reader.vectors_writer().unwrap();
    let key = "paid/synthetic/1/document/unit";
    vectors
        .put(
            key,
            &[Vector {
                text_hash: vec![9; 32],
                chunk: 0,
                values: vec![1.0],
            }],
        )
        .unwrap();
    assert!(missing(&vectors, key, &[9; 32], 1).is_empty());
    assert_eq!(vectors.load(key).unwrap().vectors(), 1);
}

#[test]
fn loaded_vectors_keep_their_database_snapshot_revision_after_a_concurrent_write() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
    texts(&archive, &[&[1], &[2]]);
    let pool = crate::ReadPool::open(&archive).unwrap();
    let writer_reader = pool.reader().unwrap();
    let key = "synthetic/model/2/document/unit";
    let mut writer = writer_reader.vectors_writer().unwrap();
    writer
        .put(
            key,
            &[Vector {
                text_hash: vec![1],
                chunk: 0,
                values: vec![1., 0.],
            }],
        )
        .unwrap();
    let old_reader = pool.reader().unwrap();
    let old = old_reader.vectors().unwrap();
    let revision = old.revision(key).unwrap();
    // A concurrent write: another text's vector.
    writer
        .put(
            key,
            &[Vector {
                text_hash: vec![2],
                chunk: 0,
                values: vec![0., 1.],
            }],
        )
        .unwrap();
    let old_load = Arc::new(old.load(key).unwrap());
    assert_eq!(old_load.revision, revision);
    assert_eq!(old_load.vectors(), 1);
    assert_eq!(old_load.similarity(&[1., 0.]).unwrap().of(&[2]), None);
    let new_reader = pool.reader().unwrap();
    let new_load = Arc::new(new_reader.vectors().unwrap().load(key).unwrap());
    assert!(new_load.revision > revision);
    assert_eq!(new_load.vectors(), 2);
    assert_eq!(new_load.similarity(&[1., 0.]).unwrap().of(&[2]), Some(0.));
}

#[test]
fn missing_chunks_keep_holes_and_dimension_keys_separate() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut vectors = reader.vectors_writer().unwrap();
    let key = "local/synthetic/2/document/unit";
    let other = "local/synthetic/3/document/unit";
    vectors
        .put(
            key,
            &[
                Vector {
                    text_hash: vec![1; 32],
                    chunk: 0,
                    values: vec![1.0, 0.0],
                },
                Vector {
                    text_hash: vec![1; 32],
                    chunk: 2,
                    values: vec![0.0, 1.0],
                },
            ],
        )
        .unwrap();
    vectors
        .put(
            other,
            &[Vector {
                text_hash: vec![1; 32],
                chunk: 1,
                values: vec![1.0, 0.0, 0.0],
            }],
        )
        .unwrap();
    assert_eq!(missing(&vectors, key, &[1; 32], 3), HashSet::from([1]));
    assert_eq!(missing(&vectors, other, &[1; 32], 3), HashSet::from([0, 2]));
    assert_eq!(
        missing(&vectors, key, &[2; 32], 3),
        HashSet::from([0, 1, 2])
    );
    let page = vectors
        .missing_chunks(key, &[(vec![1; 32], 3), (vec![2; 32], 3)])
        .unwrap();
    assert_eq!(page[&vec![1; 32]], HashSet::from([1]));
    assert_eq!(page[&vec![2; 32]], HashSet::from([0, 1, 2]));
    vectors
        .fail(key, &[2; 32], "permanent provider refusal")
        .unwrap();
    assert!(vectors.rejected(key).unwrap().contains(&vec![2; 32]));
    assert!(!vectors.rejected(other).unwrap().contains(&vec![2; 32]));
    assert_eq!(
        missing(&vectors, key, &[2; 32], 3),
        HashSet::from([0, 1, 2])
    );
}

#[test]
fn embeddings_writer_uses_full_sync() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let writer = reader.vectors_writer().unwrap();
    // 2 is FULL: a committed batch of paid vectors is on disk before put returns.
    let synchronous: i64 = writer
        .conn
        .pragma_query_value(None, "synchronous", |row| row.get(0))
        .unwrap();
    assert_eq!(synchronous, 2);
}

#[test]
fn put_refuses_different_vector() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut vectors = reader.vectors_writer().unwrap();
    let key = "paid/synthetic/2/document/unit";
    let vector = |text: u8, values: Vec<f32>| Vector {
        text_hash: vec![text; 32],
        chunk: 0,
        values,
    };
    let stored = || -> Vec<(Vec<u8>, Vec<u8>)> {
        let conn = rusqlite::Connection::open(archive.embeddings_db()).unwrap();
        conn.prepare("SELECT text_hash, vector FROM embedded ORDER BY text_hash")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    };
    vectors.put(key, &[vector(1, vec![1.0, 0.0])]).unwrap();
    let first = stored();
    // The same vector again is already stored.
    vectors.put(key, &[vector(1, vec![1.0, 0.0])]).unwrap();
    assert_eq!(stored(), first);
    // A different one for the same text, key and chunk is refused, and
    // nothing else the call carried is stored.
    let error = vectors
        .put(key, &[vector(2, vec![0.0, 1.0]), vector(1, vec![0.0, 1.0])])
        .unwrap_err();
    assert!(matches!(error, crate::StoreError::Conflict(_)), "{error}");
    assert_eq!(stored(), first);
}

/// Options for compiling a ranked query over a fixture archive.
fn ranked_options() -> crate::query::Options {
    crate::query::Options {
        clock: query_language::time::Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
                jiff::civil::time(4, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: vec!["test".into()],
        page_size: None,
        random_seed: Some(7),
        embeddings: true,
        self_session: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        chat_pages: vec![],
    }
}

/// An archive of `sessions` sessions whose answers each have one text,
/// every text with a vector from `vector(session number)`.
fn ranked_archive(
    sessions: usize,
    vector: impl Fn(usize) -> Vec<f32>,
) -> (tempfile::TempDir, crate::fixture::Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut f = crate::fixture::Fixture::new(dir.path()).unwrap();
    f.many_sessions(crate::fixture::ManySessions {
        prefix: "/home/op/work",
        sessions,
        messages: 3,
        answered: sessions,
        last: 1_800_000_000_000,
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let texts = reader
        .query(
            "SELECT hex(t.hash) AS hash, s.native_id AS native FROM text t
             JOIN block b ON b.text_id = t.id JOIN session s ON s.pk = b.session_pk",
            &[],
        )
        .unwrap();
    let vectors: Vec<Vector> = texts
        .iter()
        .map(|row| {
            let hash = row["hash"].as_str().unwrap();
            let n: usize = row["native"].as_str().unwrap()["many-".len()..]
                .parse()
                .unwrap();
            Vector {
                text_hash: (0..hash.len())
                    .step_by(2)
                    .map(|at| u8::from_str_radix(&hash[at..at + 2], 16).unwrap())
                    .collect(),
                chunk: 0,
                values: vector(n),
            }
        })
        .collect();
    assert_eq!(vectors.len(), sessions);
    reader
        .vectors_writer()
        .unwrap()
        .put("local/test/2/document/unit", &vectors)
        .unwrap();
    drop(reader);
    drop(pool);
    (dir, f)
}

/// The message IDs `text` answers in order, ranked by `similarity`.
fn ranked_ids(
    f: &crate::fixture::Fixture,
    text: &str,
    similarity: &Arc<Similarity>,
) -> Vec<String> {
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program = crate::query::compile(text, &ranked_options()).unwrap();
    assert!(program.near.is_some(), "{text}");
    let near = |_: &str| Ok(Arc::clone(similarity));
    let mut ids = Vec::new();
    reader
        .visit_query_values(&program, Some(&near), |row| {
            ids.push(row["id"].as_str().unwrap().to_owned());
            Ok::<_, crate::selection::SelectionError>(std::ops::ControlFlow::Continue(()))
        })
        .unwrap();
    ids
}

fn loaded_similarity(f: &crate::fixture::Fixture, query: &[f32]) -> Arc<Similarity> {
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let loaded = Arc::new(
        reader
            .vectors()
            .unwrap()
            .load("local/test/2/document/unit")
            .unwrap(),
    );
    Arc::new(loaded.similarity(query).unwrap())
}

#[test]
fn similar_scores_only_the_selected_rows_vectors() {
    // Session 3 alone works in d3/w3 below 1261 sessions; its answer is
    // its one text with a vector.
    for text in [
        "sessions where cwd:/home/op/work/d3/w3 similar:\"finished work\"",
        "messages where session.cwd:/home/op/work/d3/w3 similar:\"finished work\"",
    ] {
        for sessions in [100, 1_200] {
            let (_dir, f) = ranked_archive(sessions, |_| unit(&[1.0, 1.0]));
            let similarity = loaded_similarity(&f, &unit(&[1.0, 0.0]));
            let ids = ranked_ids(&f, text, &similarity);
            assert!(!ids.is_empty(), "{text}");
            assert_eq!(
                similarity.vectors_scored(),
                1,
                "{text} over {sessions} sessions"
            );
        }
    }
}

#[test]
fn the_f32_rescore_orders_what_int8_misorders() {
    // Along the query, 0.303 is nearer than 0.298, but as int8 copies
    // (each scaled by its larger value) both take 40 steps of their scale,
    // and 0.298's scale is the larger: int8 ranks it first.
    let along = |x: f32| vec![x, (1.0 - x * x).sqrt()];
    let (_dir, f) = ranked_archive(12, |n| match n {
        4 => along(0.298),
        7 => along(0.303),
        _ => vec![0.0, 1.0],
    });
    let int8 = loaded_similarity(&f, &[1.0, 0.0]);
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let hash_of = |native: &str| -> Vec<u8> {
        let hex = reader
            .query(
                "SELECT hex(t.hash) AS hash FROM text t JOIN block b ON b.text_id = t.id
                 JOIN session s ON s.pk = b.session_pk WHERE s.native_id = ?1",
                &[crate::SqlValue::Text(native.into())],
            )
            .unwrap()[0]["hash"]
            .as_str()
            .unwrap()
            .to_owned();
        (0..hex.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).unwrap())
            .collect()
    };
    let (four, seven) = (int8.of(&hash_of("many-4")), int8.of(&hash_of("many-7")));
    assert!(four.unwrap() > seven.unwrap(), "{four:?} {seven:?}");
    drop(reader);
    drop(pool);
    let similarity = loaded_similarity(&f, &[1.0, 0.0]);
    let ids = ranked_ids(&f, "messages where similar:\"near\"", &similarity);
    assert_eq!(ids[..2], ["many-7-2", "many-4-2"], "{ids:?}");
}

#[test]
fn query_vectors_are_stored_once_by_key_and_text() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
    let writer = Vectors::writer(&archive, 256).unwrap();
    let prefix = "local/synthetic/";
    assert_eq!(writer.query_vector(prefix, "parser").unwrap(), None);
    writer
        .put_query_vector("local/synthetic/2/query/unit", "parser", &[0.6, 0.8])
        .unwrap();
    // The vector stored first stays: every earlier ranking used it.
    writer
        .put_query_vector("local/synthetic/2/query/unit", "parser", &[0.8, 0.6])
        .unwrap();
    drop(writer);
    let snapshot = Vectors::snapshot(&archive, 256).unwrap();
    assert_eq!(
        snapshot.query_vector(prefix, "parser").unwrap(),
        Some(vec![0.6, 0.8])
    );
    assert_eq!(snapshot.query_vector(prefix, "parsers").unwrap(), None);
    assert_eq!(
        snapshot.query_vector("voyage/voyage-4/", "parser").unwrap(),
        None
    );
}
