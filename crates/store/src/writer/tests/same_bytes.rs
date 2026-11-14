//! An older chain's reading copied from a newer chain holding the same
//! bytes equals the reading a parse of the older chain records
//! (docs/archive.md, versioned rows), for documents too.
use super::*;
use polylogue_model::{Document, DocumentId, DocumentPurpose, DocumentRow};
use std::collections::BTreeMap;

fn document(body: &str) -> Document {
    let mut doc = Document::new(
        DocumentId::parse(&format!("claude-ai:doc-{body}")).unwrap(),
        DocumentPurpose::Context,
    );
    doc.blocks = vec![Block::Text { text: body.into() }];
    doc
}

/// Chain `number` of input `doc`, holding `fill` bytes and, when `parsed`,
/// its reading: one document named by those bytes.
fn chain(w: &TestArchive, number: u64, fill: u8, parsed: bool) {
    chain_with_member(w, number, fill, parsed, None);
}

fn chain_with_member(
    w: &TestArchive,
    number: u64,
    fill: u8,
    parsed: bool,
    member: Option<polylogue_model::PrefixMember>,
) {
    let mut batch = batch("doc", Route::Live, 0, Vec::new(), &[]);
    batch.input.format = "claude-ai".into();
    if number > 1 {
        batch.change = Change::Rewritten;
    }
    if parsed {
        batch.parse[0].output.documents.push(DocumentRow {
            offset: 0,
            end: 100,
            document: document(&(fill as char).to_string()),
        });
        batch.parse[0].output.prefix_members.extend(member);
        batch.parse[0].counts = counts(1);
    } else {
        batch.parse.clear();
    }
    w.store_reading(batch, Some(&[fill; 100])).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
}

/// Every row of the index, apart from write-counter values.
fn dump(w: &TestArchive) -> BTreeMap<String, Vec<String>> {
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let counters = [
        "admitted",
        "commit_seq",
        "changed",
        "derived_changed",
        "indexed_changed",
    ];
    tables
        .into_iter()
        .map(|table| {
            let mut statement = conn.prepare(&format!("SELECT * FROM \"{table}\"")).unwrap();
            let names: Vec<String> = statement
                .column_names()
                .into_iter()
                .map(str::to_owned)
                .collect();
            let mut rows: Vec<String> = statement
                .query_map([], |row| {
                    let mut text = String::new();
                    for (n, name) in names.iter().enumerate() {
                        if !counters.contains(&name.as_str()) {
                            let value: rusqlite::types::Value = row.get(n)?;
                            text.push_str(&format!("{name}={value:?};"));
                        }
                    }
                    Ok(text)
                })
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            // Observation IDs carry a random UUID that tells archives apart.
            for row in &mut rows {
                *row = without_uuids(row);
            }
            if table == "meta" {
                rows.retain(|row| row.starts_with("key=Text(\"building\")"));
            }
            rows.sort();
            (table, rows)
        })
        .collect()
}

/// `text` with each UUID (8-4-4-4-12 hex digits) replaced by `<uuid>`.
fn without_uuids(text: &str) -> String {
    let bytes = text.as_bytes();
    let is_uuid = |at: usize| {
        bytes.len() >= at + 36
            && bytes[at..at + 36].iter().enumerate().all(|(n, b)| {
                if [8, 13, 18, 23].contains(&n) {
                    *b == b'-'
                } else {
                    b.is_ascii_hexdigit()
                }
            })
    };
    let (mut out, mut at) = (String::new(), 0);
    while at < bytes.len() {
        if is_uuid(at) {
            out.push_str("<uuid>");
            at += 36;
        } else {
            let c = text[at..].chars().next().unwrap();
            out.push(c);
            at += c.len_utf8();
        }
    }
    out
}

#[test]
fn a_copied_document_source_equals_its_parse_and_is_not_shown() {
    // Chains 1 and 3 hold `a`, 2 and 4 hold `b`.
    let parsed = test_archive();
    for (number, fill) in [(1, b'a'), (2, b'b'), (3, b'a'), (4, b'b')] {
        chain(&parsed, number, fill, true);
    }
    let copied = test_archive();
    for (number, fill) in [(1, b'a'), (2, b'b'), (3, b'a'), (4, b'b')] {
        chain(&copied, number, fill, number > 2);
    }
    let answers = copied
        .writer
        .sender()
        .unwrap()
        .copy_readings(
            [(3, 1), (4, 2)]
                .map(|(from, to)| crate::ReadingCopy {
                    input: "doc".into(),
                    from,
                    to,
                    reading_version: "synthetic=1".into(),
                })
                .into(),
        )
        .unwrap();
    assert!(answers.iter().all(|a| matches!(a, Ok(true))), "{answers:?}");

    let (parsed_rows, copied_rows) = (dump(&parsed), dump(&copied));
    for (table, rows) in &parsed_rows {
        assert_eq!(Some(rows), copied_rows.get(table), "{table}");
    }
    for w in [&parsed, &copied] {
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        for (body, older, newer) in [("a", 1, 3), ("b", 2, 4)] {
            let id = document(body).id;
            let sources = reader
                .select_document(&id)
                .unwrap()
                .unwrap()
                .sources()
                .unwrap();
            let chains: Vec<(i64, bool)> = sources.iter().map(|s| (s.chain, s.shown)).collect();
            assert_eq!(chains, [(older, false), (newer, true)], "{body}");
            // Admission names the shown, newer chain's reading.
            let newer_admitted: Option<i64> = reader
                .query(
                    "SELECT ds.admitted AS n FROM document_source ds JOIN document d ON d.pk=ds.document_pk WHERE d.id=?1 AND ds.chain=?2",
                    &[
                        crate::SqlValue::Text(id.to_string()),
                        crate::SqlValue::Int(newer),
                    ],
                )
                .unwrap()[0]["n"]
                .as_i64();
            assert_eq!(
                reader.document_commit(&id).unwrap(),
                newer_admitted.map(|n| n as u64),
                "{body}"
            );
        }
    }
}

#[test]
fn reading_copies_preserve_portable_members_and_reparse_chain_owned_members() {
    for native in [false, true] {
        let w = test_archive();
        chain(&w, 1, b'a', false);
        let member = polylogue_model::PrefixMember {
            parent: polylogue_model::SessionId::parse("test:parent").unwrap(),
            child: polylogue_model::SessionId::parse("test:child").unwrap(),
            message: polylogue_model::MessageId::parse("test:copied").unwrap(),
            source: (!native).then(|| polylogue_model::PrefixSource {
                input: "original".into(),
                chain: 7,
                reading_version: "original=1".into(),
            }),
            position: [u64::MAX, 2],
        };
        chain_with_member(&w, 2, b'a', true, Some(member));
        let result = w
            .writer
            .sender()
            .unwrap()
            .copy_readings(vec![crate::ReadingCopy {
                input: "doc".into(),
                from: 2,
                to: 1,
                reading_version: "synthetic=1".into(),
            }])
            .unwrap();
        assert!(
            matches!(result.as_slice(), [Ok(copied)] if *copied != native),
            "{result:?}"
        );
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        let members = reader
            .query(
                "SELECT chain,source_input,source_chain,sequence FROM prefix_member ORDER BY chain",
                &[],
            )
            .unwrap();
        assert_eq!(members.len(), if native { 1 } else { 2 });
        for member in members {
            assert_eq!(
                member["source_input"],
                if native { "doc" } else { "original" }
            );
            assert_eq!(member["source_chain"], if native { 2 } else { 7 });
            assert_eq!(member["sequence"], u64::MAX.to_string());
        }
    }
}

#[cfg(feature = "faults")]
#[test]
fn stopping_a_copy_interrupts_its_insert_and_rolls_back_the_complete_reading() {
    crate::fault::clear();
    let w = test_archive();
    chain(&w, 1, b'a', false);
    chain(&w, 2, b'a', true);
    // The copy reaches SQLite's progress handler inside this INSERT SELECT,
    // after earlier tables have already copied their rows in the savepoint.
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    conn.execute_batch(
        "WITH RECURSIVE n(value) AS (VALUES(1) UNION ALL SELECT value+1 FROM n WHERE value<256)
         INSERT INTO unused(input_pk,chain,format,what,count)
         SELECT (SELECT pk FROM input_ref WHERE id='doc'),2,'synthetic','field:'||value,1 FROM n",
    )
    .unwrap();
    drop(conn);
    let before = dump(&w);
    let copy = crate::ReadingCopy {
        input: "doc".into(),
        from: 2,
        to: 1,
        reading_version: "synthetic=1".into(),
    };
    crate::fault::arm(
        "write.copy_table",
        "unused",
        crate::fault::Action::Fail,
        0,
        1,
    );
    let failed = w
        .writer
        .sender()
        .unwrap()
        .copy_readings(vec![copy.clone()])
        .unwrap();
    assert!(
        matches!(failed.as_slice(), [Err(StoreError::Io(_))]),
        "{failed:?}"
    );
    assert_eq!(
        dump(&w),
        before,
        "an ordinary copy failure retains its cause and rolls back its savepoint"
    );
    crate::fault::clear();
    crate::fault::arm(
        "write.copy_table",
        "unused",
        crate::fault::Action::Hold,
        0,
        1,
    );
    let sender = w.writer.sender().unwrap();
    let (done, result) = sync_channel(1);
    let worker = std::thread::spawn({
        let copy = copy.clone();
        move || done.send(sender.copy_readings(vec![copy])).unwrap()
    });
    let started = std::time::Instant::now();
    while !crate::fault::held() {
        assert!(started.elapsed() < std::time::Duration::from_secs(30));
        std::thread::yield_now();
    }
    // Admit a user write while the copy owns the writer. Its durable
    // archive transaction must still run after the index copy rolls back.
    let request = crate::fixture::authored::operator_request(
        crate::user_writes::new_request_id(),
        polylogue_model::user::UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
    );
    let (reply, admitted) = sync_channel(1);
    w.writer
        .sender()
        .unwrap()
        .0
        .send_user(UserMsg::Request(
            Box::new(request.clone()),
            None,
            None,
            None,
            reply,
        ))
        .unwrap();
    w.writer.stop_handle().request();
    crate::fault::release();
    assert!(matches!(
        result
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        Err(StoreError::WriteCancelled)
    ));
    worker.join().unwrap();
    crate::fault::clear();
    assert_eq!(
        dump(&w),
        before,
        "the whole transaction, including earlier copied tables, rolls back"
    );

    let first = admitted
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let repeat = w.writer.user(request).unwrap();
    assert_eq!(repeat.id, first.id);
    assert!(repeat.repeat && !repeat.pending);

    let TestArchive {
        writer,
        archive,
        _lock,
        _dir,
    } = w;
    drop(writer);
    let writer = Writer::start(&archive).unwrap();
    let copied = writer.sender().unwrap().copy_readings(vec![copy]).unwrap();
    assert!(matches!(copied.as_slice(), [Ok(true)]), "{copied:?}");
    let reads = ReadPool::open(&archive).unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader
            .query(
                "SELECT chain,parsed_upto,status FROM parse_state ORDER BY chain",
                &[]
            )
            .unwrap(),
        vec![
            serde_json::json!({"chain":1,"parsed_upto":100,"status":"done"}),
            serde_json::json!({"chain":2,"parsed_upto":100,"status":"done"}),
        ]
    );
    assert_eq!(
        reader
            .query(
                "SELECT chain,count(*) AS n FROM unused GROUP BY chain ORDER BY chain",
                &[]
            )
            .unwrap(),
        vec![
            serde_json::json!({"chain":1,"n":256}),
            serde_json::json!({"chain":2,"n":256}),
        ]
    );
}
