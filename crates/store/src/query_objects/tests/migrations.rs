use super::*;

const COMPONENT: &str = crate::schema::QUERY_OBJECTS;
const ARCHIVE: &str = include_str!("../../schema/archive_3.sql");
const PRIMARY: &str = include_str!("../../schema/index_8.sql");

fn cache_schema() -> String {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(COMPONENT).unwrap();
    // The SQLite parser rewrites foreign keys and references, as SourceSchema does.
    for name in ["query", "query_edge", "query_run"] {
        conn.execute_batch(&format!(
            "ALTER TABLE \"{name}\" RENAME TO \"source_{name}\""
        ))
        .unwrap();
    }
    let mut statement = conn.prepare(
        "SELECT type,name,sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END,name"
    ).unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .unwrap();
    let mut output = String::new();
    for row in rows {
        let (kind, name, sql) = row.unwrap();
        let sql = if kind == "index" {
            let declaration = format!("CREATE INDEX {name}");
            let body = sql.strip_prefix(&declaration).unwrap();
            format!("CREATE INDEX source_{name}{body}")
        } else {
            sql
        };
        output.push_str(&sql);
        output.push_str(";\n");
    }
    output
}

#[test]
fn query_component_migrations_match_native_and_sqlite_transformed_cache_shapes() {
    let archive = Connection::open_in_memory().unwrap();
    archive.execute_batch(ARCHIVE).unwrap();
    let tx = archive.unchecked_transaction().unwrap();
    crate::assertions::conversion::migrate_query_runs(&tx, Source::Ledger).unwrap();
    tx.commit().unwrap();
    assert!(crate::schema::same_schema(&archive, &[COMPONENT]).unwrap());
    let cache = Connection::open_in_memory().unwrap();
    cache.execute_batch(PRIMARY).unwrap();
    let tx = cache.unchecked_transaction().unwrap();
    crate::assertions::conversion::migrate_query_runs(&tx, Source::Cache).unwrap();
    tx.commit().unwrap();
    assert!(crate::schema::same_schema(&cache, &[&cache_schema()]).unwrap());
}

#[test]
fn query_component_migrations_preserve_unrelated_rows_in_each_source_namespace() {
    for migration in [ARCHIVE, PRIMARY] {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE retained(value TEXT PRIMARY KEY) STRICT; INSERT INTO retained VALUES ('synthetic original');").unwrap();
        conn.execute_batch(migration).unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        crate::assertions::conversion::migrate_query_runs(
            &tx,
            if migration == PRIMARY {
                Source::Cache
            } else {
                Source::Ledger
            },
        )
        .unwrap();
        tx.commit().unwrap();
        assert_eq!(
            conn.query_row("SELECT value FROM retained", [], |row| row
                .get::<_, String>(0))
                .unwrap(),
            "synthetic original"
        );
    }
}

#[test]
fn confirmed_query_components_rebuild_out_of_order_and_keep_unknown_first_run_honest() {
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("PRAGMA foreign_keys=ON; CREATE TABLE source_input(id TEXT PRIMARY KEY) STRICT; CREATE TABLE retained(value INTEGER NOT NULL) STRICT; INSERT INTO retained VALUES (7);").unwrap();
    conn.execute_batch(PRIMARY).unwrap();
    let tx = conn.unchecked_transaction().unwrap();
    crate::assertions::conversion::migrate_query_runs(&tx, Source::Cache).unwrap();
    tx.commit().unwrap();
    let object = question("sessions");
    let read_at = ChangePosition {
        generation: "synthetic-generation".into(),
        index: 4,
        archive: 3,
    };
    let later = run(&object, "later", 1, read_at.clone());
    let mut earlier = run(&object, "earlier", 10_000, read_at);
    earlier
        .binding
        .original_mut()
        .unwrap()
        .replay
        .context
        .clock
        .now = "2026-10-29T12:00:00.123456789Z".parse().unwrap();
    let tx = conn.transaction().unwrap();
    let completed = outcome(&earlier, u64::MAX);
    apply_run_outcome(&tx, &completed).unwrap();
    assert!(
        tx.query_row(
            "SELECT run IS NULL FROM source_query_run WHERE id=?1",
            [earlier.id.as_str()],
            |row| row.get::<_, bool>(0)
        )
        .unwrap()
    );

    apply_run(&tx, &later, 8, 1).unwrap();
    apply_run(&tx, &earlier, 8, 0).unwrap();
    let mut conflicting = earlier.clone();
    conflicting
        .binding
        .original_mut()
        .unwrap()
        .query_options
        .fields
        .push("forged".into());
    assert!(matches!(
        apply_run(&tx, &conflicting, 8, 0),
        Err(StoreError::Conflict(_))
    ));

    apply_object(&tx, &object).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        conn.query_row("SELECT first_run FROM source_query", [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        run_id("earlier").as_str()
    );
    let body: Vec<u8> = conn
        .query_row(
            "SELECT run FROM source_query_run WHERE id=?1",
            [earlier.id.as_str()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        crate::encoding::unpack_json::<QueryRun>(&body).unwrap(),
        earlier
    );
    assert_eq!(
        conn.query_row(
            "SELECT count FROM source_query_run WHERE id=?1",
            [earlier.id.as_str()],
            |row| row.get::<_, String>(0)
        )
        .unwrap(),
        u64::MAX.to_string()
    );
    let tx = conn.transaction().unwrap();
    set_first_run(&tx, object.id, None).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        conn.query_row("SELECT first_run FROM source_query", [], |row| row
            .get::<_, Option<String>>(0))
            .unwrap(),
        None
    );
    let tx = conn.transaction().unwrap();
    clear(&tx).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM source_query_run", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        conn.query_row("SELECT value FROM retained", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        7
    );
}
