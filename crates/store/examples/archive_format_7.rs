//! One-time operation: brings an `archive.db` of format 6 to the current
//! format (9) in place, beside its segment log. Not part of the product; it is run on a
//! reflinked copy first and once on the operator's archive, then deleted.
//!
//! `cargo run -p polylogue-store --example archive_format_7 -- <archive root>`
//!
//! The new file is created by the store's own schema code, filled from the
//! format-6 file row by row, and checked before it replaces it:
//! - every `extent` and `blob` row is carried over exactly (inputs renamed
//!   through the rename table, nothing else different), and every revision
//!   keeps its input, number, chain and length. The conversion never writes
//!   the segment log, so a revision's bytes, which these rows determine
//!   entirely, read back as they did;
//! - each table whose stored names changed holds as many rows as before;
//! - no row names a row that is missing;
//! - the file opens as this build's format (version and schema text).
//!
//! Not checked: whether the segment log is readable and its bytes match their
//! hashes. That is `polylogue verify`'s question, asked of the archive
//! itself. The chain step reads the prefixes it compares, and nothing else is
//! read from the segments. The format-6 file is kept beside the new one as
//! `archive.db.format-6`.
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
//! Format 8, in place on the new file: a read's run record keeps its reading
//! options as the caller wrote them (`show` and `hide` words, `neutral_names`,
//! one `alternate`); the step from 9 computes its hashes again.
//!
//! Format 9 renames stored names without rewriting a value (`RENAMED`):
//! `kept_result` is `saved_rows` and `annotation_slot` is
//! `annotation_unique_key`, and columns take the words for what they hold (a
//! place in an order is a `position`, a time ends in `_at`, which tool a file
//! comes from is its `origin`). The format-6 rows are copied under the new
//! names, so every later step reads and writes them under those names, and
//! each renamed table holds as many rows after as before.
//!
//! Format 10, in place: keys inside stored payloads that the wire also
//! carries say what they hold (`stored_keys`), with the hashes over those
//! payloads computed again.
use polylogue_store::archive::writer::agreement::Chains;
use polylogue_store::blobs::{ChainSource, StoredChain};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

mod stored_keys;

type Error = Box<dyn std::error::Error>;

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

    let conn = Connection::open(&new)?;
    conn.pragma_update(None, "journal_mode", "wal")?;
    polylogue_store::schema::ensure_archive(&conn)?;
    // The new file says the current format only once every step has run: a
    // file a failed step leaves says the format before, which the product
    // refuses to open as another format.
    conn.pragma_update(
        None,
        "user_version",
        polylogue_store::schema::ARCHIVE_VERSION - 1,
    )?;
    conn.execute("ATTACH DATABASE ?1 AS old", [old.to_str().ok_or("path")?])?;
    conn.pragma_update(None, "foreign_keys", false)?;
    let renamed = rename_rollouts(&conn)?;
    let context = Context {
        root: &root,
        new: &new,
    };
    println!("step\ttable\tbefore\tafter");
    // Steps run in the order of the formats they bring an archive from.
    if !STEPS.windows(2).all(|pair| pair[0].from <= pair[1].from) {
        return Err("the steps are not in the order of their formats".into());
    }
    for step in STEPS {
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
    let carried = carried_over(&conn)?;
    conn.execute("DETACH DATABASE old", [])?;
    conn.pragma_update(
        None,
        "user_version",
        polylogue_store::schema::ARCHIVE_VERSION,
    )?;
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
    println!(
        "carried over exactly\textents {}\tblobs {}\trevisions {}",
        carried.extents, carried.blobs, carried.revisions
    );
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

/// How many rows [`carried_over`] found equal in both files.
struct Carried {
    extents: i64,
    blobs: i64,
    revisions: i64,
}

/// Refuses the conversion unless the rows that place each revision's bytes
/// came over exactly: every `extent` and `blob` row of the format-6 file
/// (attached as `old`) is in the new one and the new one has no other, with
/// inputs renamed through `temp.renamed`, and every revision keeps its input,
/// number, chain and length. Names the first row that differs.
fn carried_over(conn: &Connection) -> Result<Carried, Error> {
    let renamed = |column: &str| {
        format!("coalesce((SELECT new FROM temp.renamed WHERE old = {column}), {column})")
    };
    let compare = |what: &str, old: String, new: &str| -> Result<i64, Error> {
        for (side, from, to) in [
            ("missing from", &old, new),
            ("added to", &new.to_owned(), old.as_str()),
        ] {
            let first: Option<String> = conn
                .query_row(
                    &format!("SELECT quote(t.k) FROM (SELECT * FROM ({from}) EXCEPT SELECT * FROM ({to})) t LIMIT 1"),
                    [],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(first) = first {
                return Err(format!("{what} row {side} the new file: {first}").into());
            }
        }
        Ok(conn.query_row(&format!("SELECT count(*) FROM ({new})"), [], |r| r.get(0))?)
    };
    let extents = compare(
        "extent",
        format!(
            "SELECT json_array({}, chain, offset, length, hex(blob), blob_offset, commit_seq) AS k FROM old.extent",
            renamed("input")
        ),
        "SELECT json_array(input, chain, offset, length, hex(blob), blob_offset, commit_seq) AS k FROM main.extent",
    )?;
    let blobs = compare(
        "blob",
        "SELECT json_array(hex(hash), size, stored_size, hex(storage)) AS k FROM old.blob".into(),
        "SELECT json_array(hex(hash), size, stored_size, hex(storage)) AS k FROM main.blob",
    )?;
    let revisions = compare(
        "revision",
        format!(
            "SELECT json_array({}, number, chain, length) AS k FROM old.revision",
            renamed("input")
        ),
        "SELECT json_array(input, number, chain, length) AS k FROM main.revision",
    )?;
    Ok(Carried {
        extents,
        blobs,
        revisions,
    })
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
    Step {
        from: 7,
        name: "reading options as written",
        tables: &["query_run"],
        run: read_options,
    },
    Step {
        from: 9,
        name: "stored keys say what they hold",
        tables: &[
            "client_request",
            "query",
            "query_run",
            "kept_result",
            "delivery",
            "annotation",
            "signal_event",
        ],
        run: stored_keys::step,
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

/// Column names, each as (earlier, now).
type Columns = &'static [(&'static str, &'static str)];

/// The stored names format 8 changed: each table's name in format 6 and 7,
/// its name now, and its columns whose names changed (earlier, now). Nothing
/// else about these tables changed, so their rows are copied column by
/// column under the new names.
const RENAMED: &[(&str, &str, Columns)] = &[
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
                "SELECT name FROM pragma_table_info('{table}', 'main') ORDER BY cid"
            ))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let old_columns: Vec<String> = tx
            .prepare(&format!(
                "SELECT name FROM pragma_table_info('{earlier}', 'old') ORDER BY cid"
            ))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        // A table added since starts empty.
        if old_columns.is_empty() {
            continue;
        }
        // Each column the earlier file holds is copied under its current
        // name; a column added since is left at its default.
        let (to, from): (Vec<&str>, Vec<&str>) = columns
            .iter()
            .map(|column| (column.as_str(), earlier_column(table, column)))
            .filter(|(_, earlier)| old_columns.iter().any(|old| old == earlier))
            .unzip();
        if from.len() != old_columns.len() {
            return Err(format!("table {table} lost a column; it needs its own copy").into());
        }
        let quoted = |names: &[&str]| {
            names
                .iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(",")
        };
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

/// A query run's ID, run and completion, as stored.
type QueryRunRow = (String, Option<Vec<u8>>, Option<Vec<u8>>);

/// Query runs a page at a time, in id order, so memory does not follow the
/// table: each page's rows are read whole before `visit` sees them, and
/// `visit` may update the rows it is given.
fn each_query_run(
    conn: &Connection,
    mut visit: impl FnMut(&str, Option<Vec<u8>>, Option<Vec<u8>>) -> Result<(), Error>,
) -> Result<u64, Error> {
    const PAGE: i64 = 512;
    let mut after = String::new();
    let mut visited = 0;
    loop {
        let page: Vec<QueryRunRow> = conn
            .prepare_cached(
                "SELECT id, run, completion FROM main.query_run WHERE id > ?1 ORDER BY id LIMIT ?2",
            )?
            .query_map(params![after, PAGE], |r| <(_, _, _)>::try_from(r))?
            .collect::<rusqlite::Result<_>>()?;
        let Some((last, _, _)) = page.last() else {
            return Ok(visited);
        };
        after = last.clone();
        for (id, run, completion) in page {
            visited += 1;
            visit(&id, run, completion)?;
        }
    }
}

fn unpack(bytes: &[u8]) -> Result<serde_json::Value, Error> {
    Ok(serde_json::from_slice(&zstd::decode_all(bytes)?)?)
}

/// Runs whose reading options are in format 7's shape.
fn earlier_shaped(conn: &Connection) -> Result<u64, Error> {
    let mut n = 0;
    each_query_run(conn, |_, run, _| {
        if let Some(run) = run {
            n += u64::from(rewrite(&mut unpack(&run)?, &mut 0) > 0);
        }
        Ok(())
    })?;
    Ok(n)
}

/// Format 8 records a read's options as the caller writes them. The run is
/// rewritten as JSON: its hashes, and its completion's copies, are computed
/// again by the step from 9 with the build's types, as the run then reads.
/// Nothing else holds those hashes: a run is cited by its ID, which does not
/// change.
///
/// Not carried, on purpose: a run that rendered several alternates (the
/// `reading.alternates` setting) names none, so reading it again follows
/// the setting, as a read that names none does; `latest` goes, as no
/// surface ever set it. The step prints how many runs had several
/// alternates.
fn read_options(conn: &Connection, _: &Context<'_>) -> Result<(), Error> {
    let before = earlier_shaped(conn)?;
    let (mut rewritten, mut several) = (0, 0);
    let visited = each_query_run(conn, |id, run, _| {
        let Some(run) = run else { return Ok(()) };
        let mut value = unpack(&run)?;
        if rewrite(&mut value, &mut several) == 0 {
            return Ok(());
        }
        conn.execute(
            "UPDATE main.query_run SET run = ?2 WHERE id = ?1",
            params![id, zstd::bulk::compress(&serde_json::to_vec(&value)?, 1)?],
        )?;
        rewritten += 1;
        Ok(())
    })?;
    let after = earlier_shaped(conn)?;
    println!("reading options as written\truns visited\t{visited}\t{visited}");
    println!("reading options as written\truns rewritten\t-\t{rewritten}");
    println!("reading options as written\truns in the earlier shape\t{before}\t{after}");
    println!("reading options as written\truns that rendered several alternates\t{several}\t-");
    if after != 0 {
        return Err(format!("{after} runs keep the earlier reading options").into());
    }
    Ok(())
}

/// Rewrites every read presentation's options under `value` to the shape
/// the caller writes, says how many it rewrote, and counts those that had
/// rendered several alternates in `several`.
fn rewrite(value: &mut serde_json::Value, several: &mut u64) -> usize {
    use serde_json::Value;
    let mut n = 0;
    match value {
        Value::Object(object) => {
            if object.get("kind").and_then(Value::as_str) == Some("read")
                && let Some(options) = object.get_mut("options")
                && let Some(written) = as_written(options)
            {
                if options["alternates"]
                    .as_array()
                    .is_some_and(|all| all.len() > 1)
                {
                    *several += 1;
                }
                *options = written;
                n += 1;
            }
            for (_, child) in object.iter_mut() {
                n += rewrite(child, several);
            }
        }
        Value::Array(items) => {
            for item in items {
                n += rewrite(item, several);
            }
        }
        _ => {}
    }
    n
}

/// Format 7's reading options as the words that give them, or none when
/// they are already in format 8's shape. Format 7 held each item kind
/// expanded (`tools: "full"` with `output_lines`, `inherited: true`...), its
/// resolved alternates and `for_others`; `latest` was never set. This is the
/// inverse format 7 had (`ReadOptions::show_hide_words`): its defaults did
/// not depend on the view, so a word is written exactly where a value
/// differs from its one default.
fn as_written(options: &serde_json::Value) -> Option<serde_json::Value> {
    use serde_json::{Map, Value, json};
    const EARLIER: &[&str] = &[
        "tools",
        "thinking",
        "harness",
        "output_lines",
        "inherited",
        "branches",
        "turns",
        "annotations",
        "for_others",
        "alternates",
        "spawns",
        "context_meter",
        "latest",
    ];
    let object = options.as_object()?;
    if !object.keys().any(|key| EARLIER.contains(&key.as_str())) {
        return None;
    }
    let (mut show, mut hide) = (Vec::<String>::new(), Vec::<String>::new());
    let level = |name: &str| object.get(name).and_then(Value::as_str);
    let lines = object
        .get("output_lines")
        .and_then(Value::as_u64)
        .unwrap_or(100);
    match level("tools") {
        Some("full") if lines == 0 => show.push("tools:full".into()),
        Some("full") => show.push(format!("tools:{lines}")),
        Some("headers") => show.push("tools:line".into()),
        Some("none") => hide.push("tools".into()),
        _ => {}
    }
    for name in ["thinking", "harness"] {
        match level(name) {
            Some("full") => show.push(format!("{name}:full")),
            Some("headers") => show.push(format!("{name}:line")),
            Some("none") => hide.push(name.into()),
            _ => {}
        }
    }
    let flag =
        |name: &str, default: bool| object.get(name).and_then(Value::as_bool).unwrap_or(default);
    for (field, word, default) in [
        ("inherited", "inherited", false),
        ("branches", "branches", false),
        ("annotations", "annotations", true),
        ("turns", "turns", false),
        ("context_meter", "context", false),
    ] {
        match (flag(field, default), default) {
            (true, false) => show.push(word.into()),
            (false, true) => hide.push(word.into()),
            _ => {}
        }
    }
    match object.get("spawns") {
        Some(Value::String(hidden)) if hidden == "hidden" => hide.push("spawns".into()),
        Some(Value::Object(inline)) => match inline.get("inline").map(|inline| &inline["depth"]) {
            Some(Value::Number(depth)) if depth.as_u64() == Some(0) => {
                show.push("spawns:card".into())
            }
            Some(Value::Number(depth)) => show.push(format!("spawns:{depth}")),
            _ => show.push("spawns:full".into()),
        },
        _ => {}
    }
    let mut written = Map::new();
    written.insert(
        "view".into(),
        object.get("view").cloned().unwrap_or(json!("transcript")),
    );
    if !show.is_empty() {
        written.insert("show".into(), json!(show));
    }
    if !hide.is_empty() {
        written.insert("hide".into(), json!(hide));
    }
    for key in ["around", "diff", "at"] {
        if let Some(value) = object.get(key).filter(|value| !value.is_null()) {
            written.insert(key.into(), value.clone());
        }
    }
    // The run rendered these alternates; one of them is the one the caller
    // would name, several came from the setting, which the read follows.
    if let Some([alternate]) = object
        .get("alternates")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
    {
        written.insert("alternate".into(), alternate.clone());
    }
    if flag("physical", false) {
        written.insert("physical".into(), json!(true));
    }
    if flag("for_others", false) {
        written.insert("neutral_names".into(), json!(true));
    }
    Some(Value::Object(written))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read run written in format 7's shape: its options expanded, its
    /// hashes those the earlier build computed.
    fn earlier_run() -> (serde_json::Value, serde_json::Value) {
        let options = serde_json::json!({
            "view": "context",
            "tools": "full",
            "output_lines": 5,
            "thinking": "none",
            "harness": "headers",
            "inherited": true,
            "branches": false,
            "turns": false,
            "annotations": false,
            "for_others": true,
            "alternates": ["translation:pl"],
            "spawns": {"inline": {"depth": 2}},
            "context_meter": true,
            "physical": false,
            "at": "2026-11-21T10:00:00Z",
            "latest": false
        });
        let written = serde_json::json!({
            "view": "context",
            "show": ["tools:5", "harness:line", "inherited", "context", "spawns:2"],
            "hide": ["thinking", "annotations"],
            "at": "2026-11-21T10:00:00Z",
            "alternate": "translation:pl",
            "neutral_names": true
        });
        (options, written)
    }

    /// A run of an object read as format 7 wrote it, with its read options
    /// replaced by `options`: the keys format 10 renamed are as they were.
    fn run_json(options: &serde_json::Value) -> serde_json::Value {
        use polylogue_model::query_objects::{
            ExecutionPresentation, ObjectExecutionBinding, QueryRun, RunBinding,
        };
        let hash = polylogue_model::BlobHash::of(b"run").to_string();
        let run = QueryRun {
            id: polylogue_model::RunId::parse(&format!(
                "{}-{}-{}-{}-{}",
                &hash[..8],
                &hash[8..12],
                &hash[12..16],
                &hash[16..20],
                &hash[20..32]
            ))
            .unwrap(),
            query_id: None,
            at: polylogue_model::Timestamp::from_unix_ms(1_763_000_000_000).unwrap(),
            read_position: polylogue_model::ChangePosition {
                index_instance: "synthetic".into(),
                index: 1,
                archive: 1,
            },
            machine: "synthetic-machine".into(),
            authorship: polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            },
            binding: RunBinding::OriginalObject {
                execution: Box::new(ObjectExecutionBinding {
                    reference: polylogue_model::Reference::parse("request:synthetic-request")
                        .unwrap(),
                    content_hash: polylogue_model::BlobHash::of(b"request"),
                    access_fingerprint: None,
                    versions: Default::default(),
                    presentation: ExecutionPresentation::Read {
                        options: Default::default(),
                        delivery: polylogue_model::read::ReadDelivery::Page,
                        format: polylogue_model::read::Format::Json,
                        page_budget: None,
                        direct_source: None,
                    },
                    accepted_request: None,
                }),
            },
        };
        let mut value = serde_json::to_value(run).unwrap();
        value["binding"]["execution"]["presentation"]["options"] = options.clone();
        let text = value
            .to_string()
            .replace(r#""query_id":"#, r#""question":"#)
            .replace(r#""read_position":"#, r#""read_at":"#)
            .replace(r#""index_instance":"#, r#""generation":"#)
            .replace(r#""access_fingerprint":"#, r#""scope_fingerprint":"#);
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn earlier_reading_options_become_the_words_that_give_them() {
        let (earlier, written) = earlier_run();
        assert_eq!(as_written(&earlier), Some(written.clone()));
        assert_eq!(as_written(&written), None);
        // Several alternates came from the setting; the rewritten read
        // follows the setting again.
        let several = serde_json::json!({"alternates": ["a", "b"], "view": "transcript"});
        assert_eq!(
            as_written(&several),
            Some(serde_json::json!({"view": "transcript"}))
        );
    }

    use polylogue_model::read::{Expansion, ReadOptions, SpawnDisplay, View};

    /// What format 7's options read as, by its own accessors: each item kind
    /// with its one default, whatever the view, and tool output lines only
    /// when tools are shown whole. Inline spawns to depth 0 place no inline
    /// reading, as cards do.
    type Effective = (
        Expansion,
        Expansion,
        Expansion,
        Option<u64>,
        [bool; 5],
        SpawnDisplay,
    );

    fn earlier_effective(options: &serde_json::Value) -> Effective {
        let level = |name: &str| match options.get(name).and_then(serde_json::Value::as_str) {
            Some("none") => Expansion::None,
            Some("full") => Expansion::Full,
            _ => Expansion::Collapsed,
        };
        let flag = |name: &str, default: bool| {
            options
                .get(name)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(default)
        };
        let tools = level("tools");
        let spawns = match options.get("spawns") {
            Some(serde_json::Value::String(hidden)) if hidden == "hidden" => SpawnDisplay::Hidden,
            Some(serde_json::Value::Object(inline)) => match inline["inline"]["depth"].as_u64() {
                Some(0) => SpawnDisplay::Card,
                Some(depth) => SpawnDisplay::Inline {
                    depth: Some(depth as u32),
                },
                None => SpawnDisplay::Inline { depth: None },
            },
            _ => SpawnDisplay::Card,
        };
        (
            tools,
            level("thinking"),
            level("harness"),
            (tools == Expansion::Full).then(|| {
                options
                    .get("output_lines")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(100)
            }),
            [
                flag("inherited", false),
                flag("branches", false),
                flag("annotations", true),
                flag("turns", false),
                flag("context_meter", false),
            ],
            spawns,
        )
    }

    fn written_effective(options: &ReadOptions) -> Effective {
        let tools = options.tools();
        (
            tools,
            options.thinking(),
            options.harness(),
            (tools == Expansion::Full).then(|| options.output_lines()),
            [
                options.inherited(),
                options.branches(),
                options.annotations(),
                options.turns(),
                options.context_meter(),
            ],
            options.spawns(),
        )
    }

    /// Every view and every value of format 7's fields reads the same after
    /// the step as before. Each accessor reads only the words of its own
    /// item kind, so the levels (with output lines) and the flags with the
    /// spawn forms are crossed in two groups, each over every view.
    #[test]
    fn written_words_read_as_the_earlier_options_did() {
        let levels = [None, Some("none"), Some("headers"), Some("full")];
        let spawns = [
            None,
            Some(serde_json::json!("hidden")),
            Some(serde_json::json!("card")),
            Some(serde_json::json!({"inline": {"depth": null}})),
            Some(serde_json::json!({"inline": {"depth": 0}})),
            Some(serde_json::json!({"inline": {"depth": 3}})),
        ];
        let flags = [
            "inherited",
            "branches",
            "annotations",
            "turns",
            "context_meter",
        ];
        let check = |options: serde_json::Value| {
            let written = as_written(&options).unwrap_or_else(|| options.clone());
            let read: ReadOptions = serde_json::from_value(written.clone())
                .unwrap_or_else(|e| panic!("{written}: {e}"));
            assert_eq!(
                written_effective(&read),
                earlier_effective(&options),
                "{options} became {written}"
            );
            assert_eq!(read.view.name(), options["view"].as_str().unwrap());
        };
        let mut cases = 0;
        for view in View::ALL {
            for tools in levels {
                for lines in [None, Some(0), Some(5), Some(100)] {
                    for thinking in levels {
                        for harness in levels {
                            let mut options =
                                serde_json::json!({"view": view.name(), "latest": false});
                            for (name, level) in [
                                ("tools", tools),
                                ("thinking", thinking),
                                ("harness", harness),
                            ] {
                                if let Some(level) = level {
                                    options[name] = serde_json::json!(level);
                                }
                            }
                            if let Some(lines) = lines {
                                options["output_lines"] = serde_json::json!(lines);
                            }
                            check(options);
                            cases += 1;
                        }
                    }
                }
            }
            for set in 0..(1u32 << (2 * flags.len())) {
                for spawn in &spawns {
                    let mut options = serde_json::json!({"view": view.name(), "for_others": false});
                    // Each flag absent, false or true: two bits each, 3 meaning absent.
                    let mut skip = false;
                    for (n, name) in flags.iter().enumerate() {
                        match (set >> (2 * n)) & 3 {
                            0 => options[*name] = serde_json::json!(false),
                            1 => options[*name] = serde_json::json!(true),
                            2 => {}
                            _ => skip = true,
                        }
                    }
                    if skip {
                        continue;
                    }
                    if let Some(spawn) = spawn {
                        options["spawns"] = spawn.clone();
                    }
                    check(options);
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, View::ALL.len() * (4 * 4 * 4 * 4 + 243 * 6));
    }

    /// The step rewrites an archive's read runs; with the step from 9 after
    /// it, both hashes in the row and in its completion are those of the run
    /// as it then reads.
    #[test]
    fn read_runs_are_rewritten_with_their_hashes() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open(dir.path().join("archive.db")).unwrap();
        polylogue_store::schema::ensure_archive(&conn).unwrap();
        let (earlier, written) = earlier_run();
        let run = run_json(&earlier);
        let id = run["id"].as_str().unwrap().to_owned();
        let pack = |value: &serde_json::Value| {
            zstd::bulk::compress(&serde_json::to_vec(value).unwrap(), 1).unwrap()
        };
        let old_hash = "0".repeat(64);
        let completion = serde_json::to_value(polylogue_model::query_objects::QueryRunOutcome {
            run: polylogue_model::RunId::parse(&id).unwrap(),
            binding_hash: polylogue_model::BlobHash::of(b"earlier binding"),
            run_hash: polylogue_model::BlobHash::of(b"earlier run"),
            at: polylogue_model::Timestamp::from_unix_ms(1_763_000_000_001).unwrap(),
            outcome: polylogue_model::query_objects::RunOutcome::Ok,
            count: Some(1),
            exact: true,
            next: None,
        })
        .unwrap();
        conn.execute(
            "INSERT INTO query_run(id, binding_hash, run_hash, object_ref, run, completion)
             VALUES (?1, ?2, ?2, ?3, ?4, ?5)",
            params![
                id,
                old_hash,
                run["binding"]["execution"]["reference"].as_str().unwrap(),
                pack(&run),
                pack(&completion)
            ],
        )
        .unwrap();
        assert_eq!(earlier_shaped(&conn).unwrap(), 1);
        let context = Context {
            root: dir.path(),
            new: &dir.path().join("archive.db"),
        };
        read_options(&conn, &context).unwrap();
        assert_eq!(earlier_shaped(&conn).unwrap(), 0);
        stored_keys::convert(&conn).unwrap();
        let (stored, hashes, done): (Vec<u8>, (String, String), Vec<u8>) = conn
            .query_row(
                "SELECT run, binding_hash, run_hash, completion FROM query_run WHERE id = ?1",
                [&id],
                |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?), r.get(3)?)),
            )
            .unwrap();
        let stored: polylogue_model::query_objects::QueryRun =
            serde_json::from_slice(&zstd::decode_all(&stored[..]).unwrap()).unwrap();
        let polylogue_model::query_objects::RunBinding::OriginalObject { execution } =
            &stored.binding
        else {
            panic!("an object run")
        };
        let polylogue_model::query_objects::ExecutionPresentation::Read { options, .. } =
            &execution.presentation
        else {
            panic!("a read")
        };
        assert_eq!(serde_json::to_value(options).unwrap(), written);
        assert_eq!(
            hashes,
            (
                stored.binding_hash().unwrap().to_string(),
                stored.record_hash().unwrap().to_string()
            )
        );
        let done: polylogue_model::query_objects::QueryRunOutcome =
            serde_json::from_slice(&zstd::decode_all(&done[..]).unwrap()).unwrap();
        assert_eq!(
            (done.binding_hash.to_string(), done.run_hash.to_string()),
            hashes
        );
    }
}

#[cfg(test)]
#[path = "archive_format_7/stored_names.rs"]
mod stored_names;

#[cfg(test)]
#[path = "archive_format_7/carried_over.rs"]
mod carried_over;
