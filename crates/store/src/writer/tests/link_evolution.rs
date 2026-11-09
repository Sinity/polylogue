use super::*;
use rusqlite::types::Value;

fn legacy_index(primary: bool) -> (TestArchive, Connection) {
    let w = test_archive();
    let mut index = Connection::open_in_memory().unwrap();
    ensure(&index, primary).unwrap();
    for (input, evidence) in [
        ("a-parent", "toolUseResult.agentId"),
        ("z-child", "agentId"),
    ] {
        let mut reading = batch(
            input,
            0,
            0,
            vec![row(
                1,
                text_message(input, "parent", Role::User, "retained message", None),
            )],
            &["parent", "child-a", "child-b", "unaffected"],
        );
        reading.parse[0].checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
            reading.parse[0].reading_version.clone(),
            100,
            Some(serde_json::json!({"retained":true})),
        ));
        for child in ["child-a", "child-b"] {
            reading.parse[0].output.links.push(Link {
                kind: LinkKind::Spawn,
                parent: sid("parent"),
                child: sid(child),
                at: None,
                call_id: (input == "a-parent").then(|| "toolu_0203".into()),
                evidence: evidence.into(),
            });
        }
        // Actual source custody precedes confirmed projection in either mode.
        let mut captured = w.commit_source(reading, &SOURCE_RECORD).unwrap();
        let Change::Reparse { chain } = captured.change else {
            panic!("captured chain")
        };
        let tx = index.transaction().unwrap();
        resolve::prepare_index(&tx, &mut captured, chain, false).unwrap();
        let counter = crate::archive::bump_counter(&tx).unwrap();
        rows::write_confirmed_rows(
            &tx,
            &captured,
            &captured.parse[0],
            chain,
            counter,
            None,
            Some(&fixture_receipt()),
        )
        .unwrap();
        tx.commit().unwrap();
    }
    // v6 has the same table shape. Its resolver selected the a-parent evidence.
    index
        .execute("UPDATE link SET evidence='toolUseResult.agentId'", [])
        .unwrap();
    index.pragma_update(None, "user_version", 6).unwrap();
    let vectors = Connection::open(w.archive.embeddings_db()).unwrap();
    vectors.execute("INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES(x'01','synthetic/model/1/document/unit',0,1,x'0000803f',123)", []).unwrap();
    (w, index)
}

fn ensure(index: &Connection, primary: bool) -> Result<()> {
    if primary {
        crate::schema::ensure_primary_index(index)
    } else {
        crate::schema::ensure_index(index)
    }
}

// Compare every stored value, omitting only the fields this repair owns.
fn snapshot(conn: &Connection, omit_repair: bool) -> Vec<(String, Vec<Vec<Value>>)> {
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    tables
        .into_iter()
        .map(|table| {
            let quote = |name: &str| format!("\"{}\"", name.replace('"', "\"\""));
            let mut info = conn
                .prepare(&format!("PRAGMA table_info({})", quote(&table)))
                .unwrap();
            let columns: Vec<String> = info
                .query_map([], |r| r.get::<_, String>(1))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
                .into_iter()
                .filter(|column| {
                    !omit_repair
                        || !matches!(
                            (table.as_str(), column.as_str()),
                            ("link", "evidence") | ("session", "changed")
                        )
                })
                .collect();
            let where_clause = if omit_repair && table == "meta" {
                " WHERE key<>'write_counter'"
            } else {
                ""
            };
            let sql = format!(
                "SELECT {} FROM {}{where_clause} ORDER BY {}",
                columns
                    .iter()
                    .map(|name| quote(name))
                    .collect::<Vec<_>>()
                    .join(","),
                quote(&table),
                (1..=columns.len())
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let rows = conn
                .prepare(&sql)
                .unwrap()
                .query_map([], |row| {
                    (0..columns.len())
                        .map(|i| row.get::<_, Value>(i))
                        .collect::<rusqlite::Result<Vec<_>>>()
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            (table, rows)
        })
        .collect()
}

fn counter(index: &Connection) -> i64 {
    index
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'",
            [],
            |r| r.get(0),
        )
        .unwrap()
}

#[test]
fn old_link_evidence_is_reprojected_without_replacing_custody_or_vectors() {
    for primary in [false, true] {
        let (w, index) = legacy_index(primary);
        let original = snapshot(&index, true);
        let raw = Connection::open(w.archive.archive_db()).unwrap();
        let raw_before = snapshot(&raw, false);
        let vectors = Connection::open(w.archive.embeddings_db()).unwrap();
        let vectors_before = snapshot(&vectors, false);
        let before_counter = counter(&index);
        let unaffected: i64 = index
            .query_row(
                "SELECT changed FROM session WHERE id='claude-code:unaffected'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        ensure(&index, primary).unwrap();
        assert_eq!(
            crate::schema::version(&index).unwrap(),
            crate::schema::INDEX_VERSION
        );
        assert_eq!(snapshot(&index, true), original);
        assert_eq!(counter(&index), before_counter + 1);
        assert_eq!(index.query_row("SELECT count(*) FROM link WHERE evidence='agentId' AND spawning_call='toolu_0203'", [], |r| r.get::<_, i64>(0)).unwrap(), 2);
        assert_eq!(index.query_row("SELECT count(*) FROM session WHERE id IN ('claude-code:parent','claude-code:child-a','claude-code:child-b') AND changed=?1", [before_counter + 1], |r| r.get::<_, i64>(0)).unwrap(), 3);
        assert_eq!(
            index
                .query_row(
                    "SELECT changed FROM session WHERE id='claude-code:unaffected'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            unaffected
        );
        let repaired = snapshot(&index, false);
        // An old index whose winner already selected this evidence needs no
        // write-counter change, even when the migration is actually entered.
        index.pragma_update(None, "user_version", 6).unwrap();
        ensure(&index, primary).unwrap();
        assert_eq!(snapshot(&index, false), repaired);
        ensure(&index, primary).unwrap();
        assert_eq!(snapshot(&index, false), repaired);
        assert_eq!(snapshot(&raw, false), raw_before);
        assert_eq!(snapshot(&vectors, false), vectors_before);
    }
}

#[test]
fn invalid_link_contribution_rolls_back_the_entire_index_upgrade() {
    for primary in [false, true] {
        let (_w, index) = legacy_index(primary);
        index.execute("UPDATE link_value SET body=x'00' WHERE pk=(SELECT value_pk FROM link_source WHERE link_pk=(SELECT max(pk) FROM link) LIMIT 1)", []).unwrap();
        let before = snapshot(&index, false);
        ensure(&index, primary).expect_err("invalid retained link value must refuse the upgrade");
        assert_eq!(crate::schema::version(&index).unwrap(), 6);
        assert_eq!(snapshot(&index, false), before);
    }
}
