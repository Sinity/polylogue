use super::*;

// The archive-6 row decoder must refuse the added policy table before applying rows.
mod archive_six {
    use super::super::super::to_sql;
    use rusqlite::types::Value as Sql;
    use rusqlite::{Connection, params_from_iter};
    use serde_json::Value;
    pub(super) fn apply_row(archive: &Connection, v: Value) -> crate::Result<u64> {
        let table = v["table"]
            .as_str()
            .filter(|t| TABLES.contains(t))
            .ok_or_else(|| {
                crate::StoreError::Invalid(format!("a row names no known table: {v}"))
            })?;
        let row = v["row"]
            .as_object()
            .ok_or_else(|| crate::StoreError::Invalid("a row is not an object".into()))?;
        let cols: Vec<&String> = row.keys().collect();
        if cols
            .iter()
            .any(|c| !c.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_'))
        {
            return Err(crate::StoreError::Invalid(format!(
                "bad column name in {table}"
            )));
        }
        let sql = format!(
            "INSERT OR IGNORE INTO {table}({}) VALUES ({})",
            cols.iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            (1..=cols.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let values: Vec<Sql> = row.values().map(to_sql).collect::<Result<_, _>>()?;
        let added = archive.execute(&sql, params_from_iter(values.iter()))?;
        if added == 0 {
            let matching = format!(
                "SELECT EXISTS(SELECT 1 FROM {table} WHERE {})",
                cols.iter()
                    .enumerate()
                    .map(|(i, c)| format!("{c} IS ?{}", i + 1))
                    .collect::<Vec<_>>()
                    .join(" AND ")
            );
            let same: bool =
                archive.query_row(&matching, params_from_iter(values.iter()), |r| r.get(0))?;
            if !same {
                return Err(crate::StoreError::Invalid(format!(
                    "restored {table} row conflicts with existing unrelated data"
                )));
            }
        }
        Ok(added as u64)
    }

    /// The tables a batch may restore into.
    const TABLES: &[&str] = &[
        "input",
        "revision",
        "blob",
        "extent",
        "machine",
        "observation",
        "note",
        "note_target",
        "note_relation",
        "excision",
        "vector_deletion_policy",
        "setting",
        "delivery",
        "delivery_event",
    ];
}

#[test]
fn document_policy_round_trips_as_owned_excision_rows_and_old_decoder_refuses() {
    let source = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&source).unwrap();
    let mut source = source;
    crate::schema::migrate_archive(&mut source).unwrap();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![],
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    source.execute("INSERT INTO excision(id,scope,refused_json,created_at,commit_seq) VALUES ('remove','document:claude-ai:doc:gone',?1,1,1)",[json!({"sessions":[],"publication":publication}).to_string()]).unwrap();
    source
        .execute(
            "INSERT INTO excision_document VALUES ('remove','claude-ai:doc:gone')",
            [],
        )
        .unwrap();
    let unit = Unit {
        commit_seq: 1,
        kind: "excision".into(),
        key: "remove".into(),
    };
    let batch = read(&source, &unit).unwrap();
    let bytes = batch.contents();
    let row = bytes
        .split(|b| *b == b'\n')
        .filter(|s| !s.is_empty())
        .map(|s| serde_json::from_slice::<Value>(s).unwrap())
        .find(|v| v["table"] == "excision_document")
        .unwrap();
    let error = archive_six::apply_row(&source, row).unwrap_err();
    assert!(error.to_string().contains("no known table"));
    let mut target = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&target).unwrap();
    crate::schema::migrate_archive(&mut target).unwrap();
    let descriptor = RestoreUnit {
        unit,
        batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
        bytes: Some(bytes.len() as u64),
        rows: batch.rows,
        blobs: vec![],
        excision: Some(publication),
    };
    let selection = select(&[descriptor]).unwrap();
    restore::install_fixture(&mut target, &selection, None, |_| {
        Ok(Some(std::io::Cursor::new(bytes.clone())))
    })
    .unwrap();
    assert_eq!(
        target
            .query_row("SELECT document_id FROM excision_document", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        "claude-ai:doc:gone"
    );
    let refused = crate::excise::Refused::load(&target).unwrap();
    assert!(
        refused
            .matching_documents(&["claude-ai:doc:gone".into(), "claude-ai:doc:kept".into()])
            .unwrap()
            .contains("claude-ai:doc:gone")
    );
    let before = crate::excise::recovery::current(&target).unwrap();
    target
        .execute(
            "UPDATE excision_document SET document_id='claude-ai:doc:changed'",
            [],
        )
        .unwrap();
    assert_ne!(crate::excise::recovery::current(&target).unwrap(), before);
}
