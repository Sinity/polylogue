//! A first build's full-text files hold what upkeep through the queue
//! writes, and rank the same.
use crate::archive::{cache, open_write};
use crate::search::SnippetIndex;
use crate::text_rows::{SearchFill, index_pending, insert_alone};
use rusqlite::{Connection, params};

/// Each term's document and occurrence counts in `table`, summed over the
/// files attached to `conn`.
fn vocabulary(conn: &Connection, table: &str) -> Vec<(String, i64, i64)> {
    let mut parts = Vec::new();
    for n in 0..crate::search::files::FILES {
        let vocab = format!("vocab_{n}_{table}");
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS temp.{vocab} USING fts5vocab(search_{n}, {table}, 'row')"
        ))
        .unwrap();
        parts.push(format!("SELECT term, doc, cnt FROM temp.{vocab}"));
    }
    conn.prepare(&format!(
        "SELECT term, sum(doc), sum(cnt) FROM ({}) GROUP BY term ORDER BY term",
        parts.join(" UNION ALL ")
    ))
    .unwrap()
    .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
    .unwrap()
    .collect::<rusqlite::Result<_>>()
    .unwrap()
}

fn ranked(conn: &Connection, index: SnippetIndex, expression: &str) -> Vec<(i64, f64)> {
    conn.prepare(&format!(
        "SELECT text_id, rank FROM ({}) ORDER BY rank, text_id",
        crate::search::ranked_texts(index, "?1", None, None)
    ))
    .unwrap()
    .query_map([expression], |row| Ok((row.get(0)?, row.get(1)?)))
    .unwrap()
    .collect::<rusqlite::Result<_>>()
    .unwrap()
}

/// A block holding `text`, as composition writes it.
fn block(conn: &Connection, pk: i64, kind: &str, text: i64) {
    conn.execute(
        "INSERT INTO block(pk,message_pk,session_pk,position,kind,text_id,content_group,content_start,content_length)
         VALUES (?1,1,1,?1,?2,?3,0,0,0)",
        params![pk, kind, text],
    )
    .unwrap();
}

#[test]
fn a_build_fills_the_files_upkeep_would_write_while_it_composes() {
    let dir = tempfile::tempdir().unwrap();
    let built = open_write(&dir.path().join("built.db"), cache::SETUP).unwrap();
    crate::schema::ensure_build_index(&built).unwrap();
    built.pragma_update(None, "foreign_keys", false).unwrap();
    let incremental = open_write(&dir.path().join("incremental.db"), cache::SETUP).unwrap();
    crate::schema::create_index(&incremental).unwrap();
    let inputs = [
        "amber cedar amber",
        "Łódź Straße Øresund",
        "cedar amber birch",
        "only unindexed words",
        "",
        "shrub's limbs: 𝔘𝔫𝔦𝔠𝔬𝔡𝔢",
        "birch amber birch birch",
    ];
    for (n, text) in inputs.into_iter().enumerate() {
        let id = n as i64 + 1;
        let hash = blake3::hash(text.as_bytes());
        let normalized = crate::encoding::normalized_text_hash(text);
        for conn in [&built, &incremental] {
            assert_eq!(
                insert_alone(conn, Some(id), hash.as_bytes(), &normalized, text, 0),
                id
            );
        }
    }
    // Texts 1 to 3 share one group, as a parse worker packs them.
    let group: i64 = built
        .query_row("SELECT group_id FROM text WHERE id=1", [], |row| row.get(0))
        .unwrap();
    let mut body = String::new();
    for id in 1..=3 {
        let start = body.len() as i64;
        body.push_str(inputs[id - 1]);
        built
            .execute(
                "UPDATE text SET group_id=?1,start=?2 WHERE id=?3",
                params![group, start, id as i64],
            )
            .unwrap();
    }
    built
        .execute(
            "UPDATE content_group SET body=?1 WHERE id=?2",
            params![crate::encoding::pack_bytes(body.as_bytes()).unwrap(), group],
        )
        .unwrap();
    // Text 6 was held before the build began, so only settling enters it.
    block(&built, 100, "text", 6);
    let mut fill = SearchFill::start(&built, 64 << 20).unwrap();
    // Composition: blocks follow, one of which leaves again.
    let blocks = [
        (1, "text", 1),
        (2, "tool_result", 1),
        (3, "text", 2),
        (4, "tool_call", 2),
        (5, "tool_result", 3),
        (6, "tool_result", 5),
        (7, "text", 7),
        (8, "tool_result", 7),
    ];
    for (pk, kind, text) in blocks {
        block(&built, pk, kind, text);
    }
    built.execute("DELETE FROM block WHERE pk=8", []).unwrap();
    built
        .execute(
            "UPDATE text SET indexed = b.bits
             FROM (SELECT text_id, max(bit) | min(bit) AS bits FROM (
                 SELECT text_id, iif(kind IN ('tool_result','tool_call'), 2, 1) AS bit
                 FROM block WHERE text_id IS NOT NULL) GROUP BY text_id) b
             WHERE text.id = b.text_id",
            [],
        )
        .unwrap();
    fill.settle(&built).unwrap();
    while !fill.wait(std::time::Duration::from_secs(1)).unwrap() {}
    drop(fill);
    crate::search::files::attach(&built).unwrap();

    // Upkeep queues each member text once per index.
    let members: Vec<(i64, i64)> = built
        .prepare("SELECT id, indexed FROM text WHERE indexed <> 0")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(members, [(1, 3), (2, 3), (3, 2), (5, 2), (6, 1), (7, 1)]);
    for (id, bits) in members {
        incremental
            .execute("UPDATE text SET indexed=?2 WHERE id=?1", [id, bits])
            .unwrap();
        for bit in [1, 2].into_iter().filter(|bit| bits & bit != 0) {
            incremental
                .execute(
                    "INSERT INTO search_pending(text_id,fts_table) VALUES (?1,?2)",
                    [id, bit],
                )
                .unwrap();
        }
    }
    index_pending(&incremental).unwrap();
    for table in ["text_fts", "output_fts"] {
        assert_eq!(
            vocabulary(&built, table),
            vocabulary(&incremental, table),
            "{table}"
        );
    }
    assert!(
        vocabulary(&built, "text_fts")
            .iter()
            .any(|row| *row == ("lodz".to_owned(), 1, 1))
    );
    for (index, expression) in [
        (SnippetIndex::Prose, "amber"),
        (SnippetIndex::Prose, "birch OR cedar"),
        (SnippetIndex::Output, "amber"),
        (SnippetIndex::Output, "\"cedar amber\""),
    ] {
        let found = ranked(&built, index, expression);
        assert!(!found.is_empty(), "{expression}");
        assert_eq!(
            found,
            ranked(&incremental, index, expression),
            "{expression}"
        );
    }
    // Text 7's tool block left before settling: it is in prose only.
    assert_eq!(
        ranked(&built, SnippetIndex::Output, "birch")
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        [3]
    );
}
