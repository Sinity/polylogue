//! One-time operation: brings an `archive.db` of format 6 to format 7 in
//! place, beside its segment log. Not part of the product; it is run on a
//! reflinked copy first and once on the operator's archive, then deleted.
//!
//! `cargo run -p polylogue-store --example archive_format_7 -- <archive root>`
//!
//! The format-7 file is created by the store's own schema code, filled from
//! the format-6 file row by row, and checked before it replaces it: every
//! revision's bytes read through its chain hash the same before and after.
//! The format-6 file is kept beside it as `archive.db.format-6`.
//!
//! What changes:
//! - `meta.archive_id` is written (the new file's own);
//! - a revision records when its bytes were read (`revision.seen_at`), the
//!   time of the observation that stored it;
//! - a Codex rollout input is named with its machine
//!   (`codex:rollout:<session id>@<machine>`), the machine of its first
//!   observation, in every row that names it;
//! - `chain` gets one row for each stored chain, its agreement with the
//!   earlier chains of its input found by comparing their bytes;
//! - `observation.path` goes (its bytes are kept in `path_bytes`, filled from
//!   it where they were missing), and an `appended` observation at the
//!   location the input was last observed at goes;
//! - a revision stored as `first` that continues an earlier chain is
//!   `appended`;
//! - `extent.capture_read` goes.
//!
//! Format 8 renames stored names without rewriting a value (`RENAMED`):
//! `kept_result` is `saved_rows` and `annotation_slot` is
//! `annotation_unique_key`, and columns take the words for what they hold (a
//! place in an order is a `position`, a time ends in `_at`, which tool a file
//! comes from is its `origin`). Rows are copied under the new names, and each
//! renamed table holds as many rows after as before.
use polylogue_store::archive::writer::agreement::Chains;
use polylogue_store::blobs::{ChainSource, StoredChain};
use rusqlite::{Connection, params};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

type Error = Box<dyn std::error::Error>;

/// Each revision's length and the BLAKE3 of its bytes, by input and number.
type Manifest = BTreeMap<(String, i64), (u64, String)>;

fn main() -> Result<(), Error> {
    let root = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: archive_format_7 <archive root>")?,
    );
    convert(&root)
}

/// Converts the archive at `root`, printing what each step did.
fn convert(root: &Path) -> Result<(), Error> {
    let root = root.to_path_buf();
    let old = root.join("archive.db");
    let new = root.join("archive.db.format-7");
    let kept = root.join("archive.db.format-6");
    if new.exists() || kept.exists() {
        return Err(format!("{} or {} exists already", new.display(), kept.display()).into());
    }
    {
        let conn = Connection::open(&old)?;
        let version: u32 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if version != 6 {
            return Err(format!("archive.db is format {version}, not 6").into());
        }
        // Everything in the log goes into the file before it is copied.
        conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")?;
    }
    let before = counts(&Connection::open(&old)?)?;
    let manifest_before = manifest(&root)?;

    let conn = Connection::open(&new)?;
    conn.pragma_update(None, "journal_mode", "wal")?;
    polylogue_store::schema::ensure_archive(&conn)?;
    conn.execute("ATTACH DATABASE ?1 AS old", [old.to_str().ok_or("path")?])?;
    conn.pragma_update(None, "foreign_keys", false)?;
    let renamed = rename_rollouts(&conn)?;
    let context = Context {
        root: &root,
        new: &new,
    };
    println!("step\ttable\tbefore\tafter");
    for step in STEPS.iter().filter(|step| step.from == 6) {
        let tx = conn.unchecked_transaction()?;
        (step.run)(&tx, &context)?;
        tx.commit()?;
        for table in step.tables {
            let count = |schema: &str| -> Result<String, Error> {
                let exists: bool = conn.query_row(
                    &format!("SELECT count(*) > 0 FROM {schema}.sqlite_schema WHERE type = 'table' AND name = ?1"),
                    [table],
                    |r| r.get(0),
                )?;
                Ok(if exists {
                    conn.query_row(
                        &format!("SELECT count(*) FROM {schema}.\"{table}\""),
                        [],
                        |r| r.get::<_, i64>(0),
                    )?
                    .to_string()
                } else {
                    "-".into()
                })
            };
            println!(
                "{}\t{table}\t{}\t{}",
                step.name,
                count("old")?,
                count("main")?
            );
        }
    }
    let violations: i64 =
        conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| {
            r.get(0)
        })?;
    if violations != 0 {
        return Err(format!("{violations} rows name rows that are not there").into());
    }
    conn.execute("DETACH DATABASE old", [])?;
    conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")?;
    let after = counts(&conn)?;
    // A table whose stored names changed keeps every row; only the
    // container links of dropped observations go with them.
    for (earlier, now, _) in RENAMED {
        let (was, is) = (before.get(*earlier), after.get(*now));
        if *now != "observation_container" && was != is {
            return Err(format!("{earlier} held {was:?} rows, {now} holds {is:?}").into());
        }
    }
    // The converted file opens as this build's format: its version and its
    // schema text equal those of a file this build creates.
    polylogue_store::schema::ensure_archive(&conn)?;
    drop(conn);

    std::fs::rename(&old, &kept)?;
    for suffix in ["-wal", "-shm"] {
        let mut from = old.clone().into_os_string();
        from.push(suffix);
        let _ = std::fs::remove_file(from);
    }
    std::fs::rename(&new, &old)?;
    for suffix in ["-wal", "-shm"] {
        let mut from = new.clone().into_os_string();
        from.push(suffix);
        let _ = std::fs::remove_file(from);
    }
    let expected: Manifest = manifest_before
        .into_iter()
        .map(|((input, number), value)| {
            (
                (renamed.get(&input).cloned().unwrap_or(input), number),
                value,
            )
        })
        .collect();
    let manifest_after = manifest(&root)?;
    if expected != manifest_after {
        // The format-6 file goes back; the converted one stays beside it.
        std::fs::rename(&old, &new)?;
        std::fs::rename(&kept, &old)?;
        return Err("a revision's bytes read back differently after the conversion".into());
    }
    println!("all tables\ttable\tbefore\tafter");
    for (table, n) in &after {
        let earlier = earlier_table(table);
        let name = if earlier == table {
            table.clone()
        } else {
            format!("{earlier} -> {table}")
        };
        println!(
            "all\t{name}\t{}\t{n}",
            before.get(earlier).map_or("-".into(), |n| n.to_string())
        );
    }
    for (table, n) in &before {
        let now = RENAMED
            .iter()
            .find(|(earlier, _, _)| earlier == table)
            .map_or(table.as_str(), |(_, now, _)| now);
        if !after.contains_key(now) {
            println!("all\t{table}\t{n}\t-");
        }
    }
    for (earlier, now, columns) in RENAMED {
        for (was, is) in *columns {
            println!("renamed\t{earlier}.{was}\t{now}.{is}");
        }
    }
    println!("rollout inputs named with their machine\t{}", renamed.len());
    println!("revisions read back equal\t{}", expected.len());
    Ok(())
}

/// Rows of every table.
fn counts(conn: &Connection) -> Result<BTreeMap<String, i64>, Error> {
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM main.sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut counts = BTreeMap::new();
    for table in tables {
        let n: i64 =
            conn.query_row(&format!("SELECT count(*) FROM main.\"{table}\""), [], |r| {
                r.get(0)
            })?;
        counts.insert(table, n);
    }
    Ok(counts)
}

/// Every revision's length and the BLAKE3 of its bytes, read through its
/// chain, from the archive's `archive.db`.
fn manifest(root: &Path) -> Result<Manifest, Error> {
    let db = root.join("archive.db");
    let conn = Connection::open(&db)?;
    let store = std::sync::Arc::new(polylogue_store::Archive::at(root).blobs()?);
    let source = ChainSource {
        path: db.clone(),
        table_prefix: "",
    };
    let mut out = BTreeMap::new();
    let mut statement =
        conn.prepare("SELECT input, number, chain, length FROM revision ORDER BY input, number")?;
    let rows: Vec<(String, i64, i64, i64)> = statement
        .query_map([], |r| <(_, _, _, _)>::try_from(r))?
        .collect::<rusqlite::Result<_>>()?;
    for (input, number, chain, length) in rows {
        let length = u64::try_from(length)?;
        let stored = StoredChain::load(&source, &input, chain, length)?;
        let mut reader = stored.reader(std::sync::Arc::clone(&store));
        let mut hash = blake3::Hasher::new();
        let read = std::io::copy(&mut reader, &mut hash)?;
        if read != length {
            return Err(format!("{input} revision {number} reads {read} of {length} bytes").into());
        }
        out.insert(
            (input, number),
            (length, hash.finalize().to_hex().to_string()),
        );
    }
    Ok(out)
}

/// The new name of each Codex rollout input named without its machine.
fn rename_rollouts(tx: &Connection) -> Result<BTreeMap<String, String>, Error> {
    let mut statement = tx.prepare(
        "SELECT i.id, (SELECT o.machine FROM old.observation o WHERE o.input = i.id
                       ORDER BY o.commit_seq, o.id LIMIT 1)
         FROM old.input i WHERE i.id LIKE 'codex:rollout:%' AND i.id NOT LIKE '%@%'",
    )?;
    let rows: Vec<(String, Option<String>)> = statement
        .query_map([], |r| <(_, _)>::try_from(r))?
        .collect::<rusqlite::Result<_>>()?;
    let mut renamed = BTreeMap::new();
    for (id, machine) in rows {
        let machine =
            machine.ok_or_else(|| format!("{id} has no observation naming its machine"))?;
        renamed.insert(id.clone(), format!("{id}@{machine}"));
    }
    tx.execute_batch(
        "CREATE TEMP TABLE renamed (old TEXT PRIMARY KEY, new TEXT NOT NULL) WITHOUT ROWID",
    )?;
    for (old, new) in &renamed {
        tx.execute(
            "INSERT INTO temp.renamed VALUES (?1, ?2)",
            params![old, new],
        )?;
    }
    Ok(renamed)
}

/// An input's name in the new file.
const NAME: &str = "coalesce((SELECT new FROM temp.renamed WHERE old = {}), {})";

fn name(column: &str) -> String {
    NAME.replace("{}", column)
}

/// What every step reads besides the two files.
struct Context<'a> {
    root: &'a Path,
    new: &'a Path,
}

/// One change of the format, applied to the new file from the old one
/// attached as `old`, with the tables it writes. Steps are keyed by the
/// version they bring an archive from and run in order; a later format
/// change appends its own.
struct Step {
    from: u32,
    name: &'static str,
    tables: &'static [&'static str],
    run: fn(&Connection, &Context<'_>) -> Result<(), Error>,
}

const STEPS: &[Step] = &[
    Step {
        from: 6,
        name: "archive identity",
        tables: &["meta"],
        run: identity,
    },
    Step {
        from: 6,
        name: "inputs and revisions",
        tables: &["input", "revision"],
        run: inputs_and_revisions,
    },
    Step {
        from: 6,
        name: "extents",
        tables: &["extent"],
        run: extents,
    },
    Step {
        from: 6,
        name: "observations",
        tables: &["observation", "observation_container"],
        run: observations,
    },
    Step {
        from: 6,
        name: "other tables",
        tables: &[],
        run: other_tables,
    },
    Step {
        from: 6,
        name: "chain rows",
        tables: &["chain"],
        run: chain_rows,
    },
];

/// The new file keeps its own archive identity; the write counter comes over.
fn identity(tx: &Connection, _: &Context<'_>) -> Result<(), Error> {
    tx.execute("INSERT INTO main.meta SELECT * FROM old.meta WHERE key <> 'archive_id' ON CONFLICT(key) DO UPDATE SET value = excluded.value", [])?;
    Ok(())
}

/// Inputs with their new names; a revision with the time of the observation
/// that stored it, and `appended` where it continues an earlier chain.
fn inputs_and_revisions(tx: &Connection, _: &Context<'_>) -> Result<(), Error> {
    tx.execute(
        &format!(
            "INSERT INTO main.input(id, origin) SELECT {}, format FROM old.input",
            name("id")
        ),
        [],
    )?;
    tx.execute(
        &format!(
            "INSERT INTO main.revision(input, number, produced, chain, length, seen_at, mtime_ns,
                    ctime_ns, commit_seq)
             SELECT {}, r.number,
                    CASE WHEN r.produced = 'first' AND r.chain < r.number THEN 'appended' ELSE r.produced END,
                    r.chain, r.length, s.seen_at, s.mtime_ns, s.ctime_ns, r.commit_seq
             FROM old.revision r LEFT JOIN old.observation s ON s.id = (
                 SELECT o.id FROM old.observation o WHERE o.input = r.input AND o.revision = r.number
                 ORDER BY o.id LIMIT 1)",
            name("r.input")
        ),
        [],
    )?;
    Ok(())
}

/// Extents without the capture read nothing reads.
fn extents(tx: &Connection, _: &Context<'_>) -> Result<(), Error> {
    tx.execute(
        &format!(
            "INSERT INTO main.extent(input, chain, offset, length, blob, blob_offset, commit_seq)
             SELECT {}, chain, offset, length, blob, blob_offset, commit_seq FROM old.extent",
            name("input")
        ),
        [],
    )?;
    Ok(())
}

/// Observations without their display path, and without an append at the
/// location its input was last observed at, which its revision records.
fn observations(tx: &Connection, _: &Context<'_>) -> Result<(), Error> {
    // An append observed at the location its input was last observed at
    // records nothing the revision does not.
    tx.execute(
        &format!(
            "INSERT INTO main.observation(id, input, machine, class, revision, chain, length,
                device, inode, size, mtime_ns, ctime_ns, details_json, polylogue_version,
                seen_at, commit_seq, path_bytes)
             SELECT o.id, {}, o.machine, o.class, o.revision, o.chain, o.length, o.device, o.inode,
                o.size, o.mtime_ns, o.ctime_ns, o.details_json, o.polylogue_version, o.seen_at,
                o.commit_seq, coalesce(o.path_bytes, CAST(o.path AS BLOB))
             FROM old.observation o
             WHERE NOT (o.class = 'appended' AND EXISTS (
                SELECT 1 FROM old.observation p WHERE p.input = o.input AND p.id < o.id
                  AND p.machine = o.machine AND p.class <> 'deleted'
                  AND p.device IS o.device AND p.inode IS o.inode
                  AND coalesce(p.path_bytes, CAST(p.path AS BLOB)) = coalesce(o.path_bytes, CAST(o.path AS BLOB))
                  AND NOT EXISTS (SELECT 1 FROM old.observation q WHERE q.input = o.input
                                   AND q.id > p.id AND q.id < o.id)))",
            name("o.input")
        ),
        [],
    )?;
    // Every dropped observation leaves its time and file times on the
    // revision it stored.
    let dropped: i64 = tx.query_row(
        "SELECT count(*) FROM old.observation o WHERE o.id NOT IN (SELECT id FROM main.observation)",
        [],
        |r| r.get(0),
    )?;
    let kept: i64 = tx.query_row(
        &format!(
            "SELECT count(*) FROM old.observation o JOIN main.revision r ON r.input = {}
               AND r.number = o.revision AND r.seen_at = o.seen_at
               AND r.mtime_ns IS o.mtime_ns AND r.ctime_ns IS o.ctime_ns
             WHERE o.id NOT IN (SELECT id FROM main.observation)",
            name("o.input")
        ),
        [],
        |r| r.get(0),
    )?;
    println!("observations\tdropped appends at an unchanged location\t{dropped}\t-");
    println!("observations\trevisions holding a dropped append's times\t-\t{kept}");
    if dropped != kept {
        return Err(
            format!("{dropped} observations dropped, {kept} revisions hold their times").into(),
        );
    }
    tx.execute(
        &format!(
            "INSERT INTO main.observation_container(observation_id, container_input, member_position, kind, path_bytes)
             SELECT c.observation_id, {}, c.member_index, c.kind, c.path_bytes
             FROM old.observation_container c
             WHERE EXISTS (SELECT 1 FROM main.observation o WHERE o.id = c.observation_id)",
            name("c.container_input")
        ),
        [],
    )?;
    Ok(())
}

/// The stored names format 8 changed: each table's name in format 6 and 7,
/// its name now, and its columns whose names changed (earlier, now). Nothing
/// else about these tables changed, so their rows are copied column by
/// column under the new names.
const RENAMED: &[(&str, &str, &[(&str, &str)])] = &[
    ("machine", "machine", &[("first_seen", "first_seen_at")]),
    ("input", "input", &[("format", "origin")]),
    ("kept_result", "saved_rows", &[("read_at", "read_position")]),
    ("delivery_event", "delivery_event", &[("what", "outcome")]),
    (
        "chat_action_event",
        "chat_action_event",
        &[("n", "position")],
    ),
    (
        "import_job_event",
        "import_job_event",
        &[("n", "position"), ("position", "provider_position")],
    ),
    (
        "observation_container",
        "observation_container",
        &[("member_index", "member_position")],
    ),
    (
        "annotation_link",
        "annotation_link",
        &[("record", "annotation")],
    ),
    (
        "annotation_slot",
        "annotation_unique_key",
        &[("slot", "key_hash")],
    ),
    (
        "alternate_part",
        "alternate_part",
        &[
            ("n", "position"),
            ("of", "source"),
            ("of_hash", "source_hash"),
        ],
    ),
    ("query_edge", "query_edge", &[("how", "kind")]),
    (
        "signal_event",
        "signal_event",
        &[("ordinal", "request_position")],
    ),
    ("row_count", "row_count", &[("n", "count")]),
    ("query_run", "query_run", &[("read_at", "read_position")]),
];

/// A table's name in the format-6 file.
fn earlier_table(table: &str) -> &str {
    RENAMED
        .iter()
        .find(|(_, now, _)| *now == table)
        .map_or(table, |(earlier, _, _)| earlier)
}

/// A column's name in the format-6 file.
fn earlier_column<'a>(table: &str, column: &'a str) -> &'a str {
    RENAMED
        .iter()
        .find(|(_, now, _)| *now == table)
        .and_then(|(_, _, columns)| columns.iter().find(|(_, now)| *now == column))
        .map_or(column, |(earlier, _)| earlier)
}

/// Every table this change leaves as it is, copied as it is, under the
/// names format 8 gives it.
fn other_tables(tx: &Connection, _: &Context<'_>) -> Result<(), Error> {
    let changed = [
        "meta",
        "input",
        "revision",
        "extent",
        "observation",
        "observation_container",
        "chain",
        "row_count",
        "annotation_import",
    ];
    let tables: Vec<String> = tx
        .prepare("SELECT name FROM main.sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for table in tables
        .iter()
        .filter(|table| !changed.contains(&table.as_str()))
    {
        let earlier = earlier_table(table);
        let columns: Vec<String> = tx
            .prepare(&format!(
                "SELECT name FROM main.pragma_table_info('{table}') ORDER BY cid"
            ))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let old_columns: Vec<String> = tx
            .prepare(&format!(
                "SELECT name FROM old.pragma_table_info('{earlier}') ORDER BY cid"
            ))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let from: Vec<&str> = columns
            .iter()
            .map(|column| earlier_column(table, column))
            .collect();
        if from != old_columns {
            return Err(format!("table {table} changed shape; it needs its own copy").into());
        }
        let quoted = |names: &[&str]| {
            names
                .iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(",")
        };
        let to: Vec<&str> = columns.iter().map(String::as_str).collect();
        tx.execute(
            &format!(
                "INSERT INTO main.\"{table}\"({}) SELECT {} FROM old.\"{earlier}\"",
                quoted(&to),
                quoted(&from)
            ),
            [],
        )?;
    }
    // Rows naming a renamed input are renamed with it, after they are copied.
    tx.execute(
        "INSERT INTO main.annotation_import SELECT * FROM old.annotation_import",
        [],
    )?;
    tx.execute(
        "UPDATE main.annotation_import SET input = (SELECT new FROM temp.renamed WHERE old = input)
         WHERE input IN (SELECT old FROM temp.renamed)",
        [],
    )?;
    Ok(())
}

/// One `chain` row for every stored chain, in commit order, its agreement
/// with the earlier chains of its input found by comparing bytes as the
/// writer does when a chain is stored.
fn chain_rows(conn: &Connection, context: &Context<'_>) -> Result<(), Error> {
    let (root, db) = (context.root, context.new);
    let store = polylogue_store::Archive::at(root).blobs()?;
    let source = ChainSource {
        path: db.to_path_buf(),
        table_prefix: "",
    };
    let chains: Vec<(String, i64, i64)> = conn
        .prepare(
            "SELECT input, chain, min(commit_seq) FROM revision WHERE number = chain
             GROUP BY input, chain ORDER BY input, chain",
        )?
        .query_map([], |r| <(_, _, _)>::try_from(r))?
        .collect::<rusqlite::Result<_>>()?;
    let length = |input: &str, chain: i64| -> Result<u64, Error> {
        let n: i64 = conn.query_row(
            "SELECT max(length) FROM revision WHERE input = ?1 AND chain = ?2",
            params![input, chain],
            |r| r.get(0),
        )?;
        Ok(u64::try_from(n)?)
    };
    let tx = conn;
    let mut recorded = Chains::default();
    let mut current = None::<String>;
    for (input, chain, commit_seq) in chains {
        if current.as_deref() != Some(input.as_str()) {
            recorded = Chains::default();
            current = Some(input.clone());
        }
        let this = StoredChain::load(&source, &input, chain, length(&input, chain)?)?;
        let latest = recorded.0.keys().next_back().copied();
        let agreement = recorded.agreement(latest, |other, from| {
            let n: i64 = conn.query_row(
                "SELECT max(length) FROM revision WHERE input = ?1 AND chain = ?2",
                params![input, other],
                |r| r.get(0),
            )?;
            let other = StoredChain::load(&source, &input, other, n.max(0) as u64)?;
            this.shared_prefix(&other, &store, from)
        })?;
        tx.execute(
            "INSERT INTO chain(input, number, begins_like, agrees_to, commit_seq) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![input, chain, agreement.begins_like, agreement.agrees_to as i64, commit_seq],
        )?;
        recorded.0.insert(chain, agreement);
    }
    Ok(())
}

#[cfg(test)]
#[path = "archive_format_7/stored_names.rs"]
mod stored_names;
