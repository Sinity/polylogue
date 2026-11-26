//! An older chain's reading copied from a newer chain holding the same
//! bytes equals the reading a parse of the older chain records
//! (docs/archive.md, versioned rows), for documents too.
mod occurrences;

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
        batch.parse[0].counts.wire_shapes.insert(
            serde_json::to_string(&("document", "/body")).unwrap(),
            polylogue_model::WireShape {
                record_kind: "document".into(),
                path: "/body".into(),
                types: BTreeMap::from([("string".into(), 1)]),
                expected: vec!["string".into()],
                dynamic_map: false,
                known_kinds: vec!["document".into()],
                unknown_kind_count: 0,
                unused_count: 0,
                table: None,
                unknown_table_count: 0,
            },
        );
    } else {
        batch.parse.clear();
    }
    w.store_reading(batch, Some(&[fill; 100])).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
}

#[test]
fn chain_layouts_report_missing_and_present_unread_readings() {
    for parsed in [false, true] {
        let w = test_archive();
        chain(&w, 1, b'a', parsed);
        chain(&w, 2, b'b', parsed);
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        let layouts = reader.chain_layouts("doc").unwrap().unwrap();
        assert_eq!(layouts.len(), 2);
        assert!(!layouts.unread_before(0).unwrap());
        assert_eq!(layouts.unread_before(1).unwrap(), !parsed);
        assert_eq!(layouts.unread_before(2).unwrap(), !parsed);
        for n in 0..2 {
            assert_eq!(layouts.get(n).unwrap().reading.is_some(), parsed);
        }
    }
}

/// Every row of the index, apart from write-counter values.
fn dump(w: &TestArchive) -> BTreeMap<String, Vec<String>> {
    index_rows(w, false)
}

fn index_rows(w: &TestArchive, exact: bool) -> BTreeMap<String, Vec<String>> {
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let counters = [
        "committed",
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
                        if exact || !counters.contains(&name.as_str()) {
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
            if !exact {
                for row in &mut rows {
                    *row = without_uuids(row);
                }
            }
            if !exact && table == "meta" {
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
    for (from, to) in [(3, 1), (4, 2)] {
        assert!(
            copied
                .writer
                .copy_reading(crate::ReadingCopy {
                    input: "doc".into(),
                    from,
                    to,
                    parse_version: "synthetic=1".into(),
                    upto: 100,
                    reaches_end: true,
                })
                .unwrap()
                .unwrap()
        );
    }

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
            // The commit position names the shown, newer chain's reading.
            let newer_committed: Option<i64> = reader
                .query(
                    "SELECT ds.committed AS n FROM document_parse ds WHERE ds.document_id=?1 AND ds.chain=?2",
                    &[
                        crate::SqlValue::Text(id.to_string()),
                        crate::SqlValue::Int(newer),
                    ],
                )
                .unwrap()[0]["n"]
                .as_i64();
            assert_eq!(
                reader.document_commit(&id).unwrap(),
                newer_committed.map(|n| n as u64),
                "{body}"
            );
        }
    }
}

#[test]
fn reading_copies_keep_portable_members_and_reparse_chain_members() {
    for unsourced in [false, true] {
        let w = test_archive();
        chain(&w, 1, b'a', false);
        let member = polylogue_model::PrefixMember {
            parent: polylogue_model::SessionId::parse("test:parent").unwrap(),
            child: polylogue_model::SessionId::parse("test:child").unwrap(),
            message: polylogue_model::MessageId::parse("test:copied").unwrap(),
            source: (!unsourced).then(|| polylogue_model::PrefixSource {
                input: "original".into(),
                chain: 7,
                parse_version: "original=1".into(),
            }),
            position: [u64::MAX, 2],
        };
        chain_with_member(&w, 2, b'a', true, Some(member));
        let before = index_rows(&w, true);
        let result = w
            .writer
            .copy_reading(crate::ReadingCopy {
                input: "doc".into(),
                from: 2,
                to: 1,
                parse_version: "synthetic=1".into(),
                upto: 100,
                reaches_end: true,
            })
            .unwrap();
        assert!(
            matches!(result, Ok(copied) if copied != unsourced),
            "{result:?}"
        );
        if unsourced {
            assert_eq!(
                index_rows(&w, true),
                before,
                "a refused copy changes nothing"
            );
        }
        let pool = ReadPool::open(&w.archive).unwrap();
        let reader = pool.reader().unwrap();
        let members = reader
            .query(
                "SELECT chain,source_input,source_chain,sequence FROM prefix_member ORDER BY chain",
                &[],
            )
            .unwrap();
        assert_eq!(members.len(), if unsourced { 1 } else { 2 });
        for member in members {
            assert_eq!(
                member["source_input"],
                if unsourced { "doc" } else { "original" }
            );
            assert_eq!(member["source_chain"], if unsourced { 2 } else { 7 });
            assert_eq!(member["sequence"], u64::MAX.to_string());
        }
    }
}

#[cfg(feature = "faults")]
#[test]
fn stopping_a_later_copy_keeps_the_completed_chain_and_rolls_back_its_own_reading() {
    crate::fault::clear();
    let w = test_archive();
    chain(&w, 1, b'a', false);
    chain(&w, 2, b'a', true);
    chain(&w, 3, b'b', false);
    chain(&w, 4, b'b', true);
    let copies = [(2, 1), (4, 3)].map(|(from, to)| crate::ReadingCopy {
        input: "doc".into(),
        from,
        to,
        parse_version: "synthetic=1".into(),
        upto: 100,
        reaches_end: true,
    });
    // The later copy stops inside the last table's INSERT SELECT. All
    // preceding tables, including its cursor and wire occurrences, have copied.
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    conn.execute_batch(
        "WITH RECURSIVE n(value) AS (VALUES(1) UNION ALL SELECT value+1 FROM n WHERE value<256)
         INSERT INTO prefix_member(input_pk,chain,parent_id,child_id,message_id,source_input,source_chain,source_version,sequence,position)
         SELECT (SELECT pk FROM input_ref WHERE id='doc'),4,'parent','child','message:'||value,'original',7,'original=1','1','2' FROM n"
    ).unwrap();
    drop(conn);
    let sender = w.writer.sender();
    assert!(sender.copy_reading(copies[0].clone()).unwrap().unwrap());
    let completed = index_rows(&w, true);
    let reads = ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    let behind = || {
        reader
            .chains_behind_index(
                &[],
                crate::StoppedChains::Finished,
                crate::RevisionChains::Older,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|chain| chain.progress.chain)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        behind(),
        vec![3],
        "the committed first chain is no longer selected"
    );
    let occurrences = reader
        .query("SELECT occurrence FROM wire_occurrence WHERE chain=1", &[])
        .unwrap();
    assert!(
        !occurrences.is_empty(),
        "the completed copy retakes real source commits"
    );
    let source = rusqlite::Connection::open(w.archive.archive_db()).unwrap();
    let expected_commit = crate::source::Source::reading_commit(&source, "doc", 1, 100).unwrap();
    drop(source);
    for row in occurrences {
        let occurrence: serde_json::Value =
            serde_json::from_str(row["occurrence"].as_str().unwrap()).unwrap();
        assert_eq!(occurrence["input"], "doc");
        assert_eq!(occurrence["chain"], 1);
        assert_eq!(occurrence["sequence"], expected_commit.sequence);
        assert_eq!(occurrence["id"], serde_json::json!(expected_commit.id));
        assert_eq!(
            occurrence["observation"],
            serde_json::json!(expected_commit.observation)
        );
        assert_eq!(
            occurrence["observed_at_ms"],
            serde_json::json!(expected_commit.observed_at_ms)
        );
    }
    drop(reader);
    drop(reads);
    crate::fault::arm(
        "write.copy_table",
        "prefix_member",
        crate::fault::Action::Hold,
        0,
        1,
    );
    let (done, result) = sync_channel(1);
    let second = copies[1].clone();
    let worker = std::thread::spawn(move || done.send(sender.copy_reading(second)).unwrap());
    crate::fault::wait_held();
    let while_copying = index_rows(&w, true);
    // Already-accepted authored work stays stored while rebuildable copy
    // work rolls back.
    let request = crate::fixture::authored::operator_request(
        crate::user_writes::new_request_id(),
        polylogue_model::user::UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
    );
    let crate::writer::Queued::Committing(accepted) = w
        .writer
        .archive()
        .try_submit(crate::writer::UserSubmission {
            request: request.clone(),
            scope: None,
            preparation: None,
            parent: None,
        })
        .unwrap()
    else {
        panic!("the user queue has room")
    };
    w.archive.stop().cancel();
    crate::fault::release();
    assert!(matches!(
        result
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap(),
        Err(StoreError::Stopped)
    ));
    worker.join().unwrap();
    crate::fault::clear();
    assert_eq!(
        while_copying, completed,
        "the in-progress copy is invisible"
    );
    assert_eq!(
        index_rows(&w, true),
        completed,
        "every copied table, occurrence, cursor and counter of only the interrupted chain rolls back"
    );
    let first = accepted
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let repeat = w.writer.archive().user(request).unwrap();
    assert_eq!(repeat.id, first.outcome.outcome.id);
    assert!(repeat.repeat && !repeat.pending);

    let TestArchive {
        writer,
        archive,
        _lock,
        _dir,
    } = w;
    drop(writer);
    let archive = archive.reopened();
    let writer = Writer::start(&archive).unwrap();
    let reads = ReadPool::open(&archive).unwrap();
    let reader = reads.reader().unwrap();
    let remaining = reader
        .chains_behind_index(
            &[],
            crate::StoppedChains::Finished,
            crate::RevisionChains::Older,
            None,
        )
        .unwrap();
    assert_eq!(
        remaining
            .iter()
            .map(|chain| chain.progress.chain)
            .collect::<Vec<_>>(),
        vec![3]
    );
    drop(reader);
    assert!(writer.copy_reading(copies[1].clone()).unwrap().unwrap());
    let reader = reads.reader().unwrap();
    let committed = reader
        .query("SELECT value FROM meta WHERE key='write_counter'", &[])
        .unwrap();
    drop(reader);
    assert!(!writer.copy_reading(copies[1].clone()).unwrap().unwrap());
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT value FROM meta WHERE key='write_counter'", &[])
            .unwrap(),
        committed,
        "a retry refuses the already-recorded chain without a write"
    );
    assert!(
        reader
            .chains_behind_index(
                &[],
                crate::StoppedChains::Finished,
                crate::RevisionChains::Older,
                None
            )
            .unwrap()
            .is_empty()
    );
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
            serde_json::json!({"chain":3,"parsed_upto":100,"status":"done"}),
            serde_json::json!({"chain":4,"parsed_upto":100,"status":"done"}),
        ]
    );
    assert_eq!(
        reader
            .query(
                "SELECT chain,count(*) AS n FROM prefix_member GROUP BY chain ORDER BY chain",
                &[]
            )
            .unwrap(),
        vec![
            serde_json::json!({"chain":3,"n":256}),
            serde_json::json!({"chain":4,"n":256}),
        ]
    );
}

#[cfg(feature = "faults")]
#[test]
fn copy_failure_keeps_its_cause_and_rolls_back_its_transaction() {
    crate::fault::clear();
    let w = test_archive();
    chain(&w, 1, b'a', false);
    chain(&w, 2, b'a', true);
    let before = index_rows(&w, true);
    crate::fault::arm(
        "write.copy_table",
        "unused",
        crate::fault::Action::Fail,
        0,
        1,
    );
    let answer = w
        .writer
        .copy_reading(crate::ReadingCopy {
            input: "doc".into(),
            from: 2,
            to: 1,
            parse_version: "synthetic=1".into(),
            upto: 100,
            reaches_end: true,
        })
        .unwrap();
    crate::fault::clear();
    assert!(matches!(answer, Err(StoreError::Io(_))), "{answer:?}");
    assert_eq!(index_rows(&w, true), before);
}
