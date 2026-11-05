//! Synthetic archives in the older format for tests: the older format's own
//! declaration (`archive.sql`, `embeddings.sql`) with rows a test writes.
//! Bytes are stored as the older builds stored them: inline as one zstd frame,
//! or as a file of zstd frames followed by a zstd seekable-format seek table.

use super::{ARCHIVE_SQL, EMBEDDINGS_SQL};
use polylogue_model::BlobHash;
use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};

/// Every file under `root` with the BLAKE3 of its bytes, to show a folder
/// was not written.
pub fn file_hashes(root: &Path) -> std::collections::BTreeMap<PathBuf, String> {
    let mut out = std::collections::BTreeMap::new();
    let mut folders = vec![root.to_path_buf()];
    while let Some(folder) = folders.pop() {
        for entry in std::fs::read_dir(&folder).expect("folder") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                folders.push(path);
            } else {
                let bytes = std::fs::read(&path).expect("file");
                out.insert(path, blake3::hash(&bytes).to_hex().to_string());
            }
        }
    }
    out
}

pub struct Builder {
    pub root: PathBuf,
    pub archive: Connection,
    embeddings: Option<Connection>,
}

/// The skippable frame holding the seek table, and the seek table footer.
const SEEK_TABLE_FRAME: u32 = 0x184D_2A5E;
const SEEKABLE_FOOTER: u32 = 0x8F92_EAB1;

/// `bytes` as the older blob files held them: one zstd frame per `frame`
/// bytes, each with its content checksum, then the seek table.
pub fn seekable(bytes: &[u8], frame: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut table = Vec::new();
    let chunks: Vec<&[u8]> = if bytes.is_empty() {
        vec![bytes]
    } else {
        bytes.chunks(frame.max(1)).collect()
    };
    for chunk in &chunks {
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 3).expect("encoder");
        encoder.include_checksum(true).expect("checksum flag");
        std::io::Write::write_all(&mut encoder, chunk).expect("compress");
        let compressed = encoder.finish().expect("frame");
        table.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        table.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(&compressed);
    }
    table.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    table.push(0);
    table.extend_from_slice(&SEEKABLE_FOOTER.to_le_bytes());
    out.extend_from_slice(&SEEK_TABLE_FRAME.to_le_bytes());
    out.extend_from_slice(&(table.len() as u32).to_le_bytes());
    out.extend_from_slice(&table);
    out
}

impl Builder {
    /// A new older archive at `version` in `root`. Versions below 4 lack
    /// what the later versions added.
    pub fn create(root: &Path, version: u32) -> Self {
        std::fs::create_dir_all(root.join("blobs")).expect("blobs folder");
        let archive = Connection::open(root.join("archive.db")).expect("archive.db");
        archive
            .execute_batch(ARCHIVE_SQL)
            .expect("older declaration");
        if version < 4 {
            archive
                .execute_batch(
                    "DROP TRIGGER input_added; DROP TRIGGER input_removed;
                     DROP TRIGGER revision_added; DROP TRIGGER revision_removed;
                     DROP TRIGGER blob_added; DROP TRIGGER blob_changed; DROP TRIGGER blob_removed;
                     DROP TABLE row_count;",
                )
                .expect("version 3");
        }
        if version < 3 {
            archive
                .execute_batch(
                    "ALTER TABLE client_request DROP COLUMN request;
                     ALTER TABLE client_request DROP COLUMN commit_seq;",
                )
                .expect("version 2");
        }
        if version < 2 {
            archive
                .execute_batch("DROP TABLE backfill_job_event; DROP TABLE backfill_job;")
                .expect("version 1");
        }
        archive
            .pragma_update(None, "user_version", version)
            .expect("version");
        Self {
            root: root.to_path_buf(),
            archive,
            embeddings: None,
        }
    }

    pub fn machine(&self, id: &str, host: &str, first_seen: i64) {
        self.archive
            .execute(
                "INSERT INTO machine(id, host, first_seen) VALUES (?1, ?2, ?3)",
                params![id, host, first_seen],
            )
            .expect("machine");
    }

    /// Stores `bytes` inline, as one zstd frame in `blob.data`.
    pub fn inline_blob(&self, bytes: &[u8]) -> BlobHash {
        let hash = BlobHash::of(bytes);
        let data = zstd::bulk::compress(bytes, 3).expect("compress");
        self.archive
            .execute(
                "INSERT INTO blob(hash, size, stored_size, place, encoding, state, data)
                 VALUES (?1, ?2, ?3, 'inline', 'plain', 'present', ?4)",
                params![&hash.0[..], bytes.len() as i64, data.len() as i64, data],
            )
            .expect("inline blob");
        hash
    }

    /// Stores `bytes` as a file under `blobs/`, in frames of `frame` bytes.
    pub fn file_blob(&self, bytes: &[u8], frame: usize) -> BlobHash {
        let hash = BlobHash::of(bytes);
        let file = seekable(bytes, frame);
        let path = self.blob_path(&hash);
        std::fs::create_dir_all(path.parent().expect("shard")).expect("shard folder");
        std::fs::write(&path, &file).expect("blob file");
        self.archive
            .execute(
                "INSERT INTO blob(hash, size, stored_size, place, encoding, state, data)
                 VALUES (?1, ?2, ?3, 'file', 'plain', 'present', NULL)",
                params![&hash.0[..], bytes.len() as i64, file.len() as i64],
            )
            .expect("file blob");
        hash
    }

    pub fn blob_path(&self, hash: &BlobHash) -> PathBuf {
        let hex = hash.to_string();
        self.root
            .join("blobs")
            .join(&hex[..2])
            .join(format!("{hex}.zst"))
    }

    /// A client request, its body packed as the older builds packed it.
    pub fn request(&self, id: &str, created_at: i64, body: &serde_json::Value, commit_seq: i64) {
        let text = serde_json::to_string(body).expect("request");
        let packed = zstd::bulk::compress(text.as_bytes(), 3).expect("compress");
        self.archive
            .execute(
                "INSERT INTO client_request(id, content_hash, created_at, request, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    id,
                    blake3::hash(text.as_bytes()).as_bytes().as_slice(),
                    created_at,
                    packed,
                    commit_seq
                ],
            )
            .expect("request");
    }

    /// A delivery row with its text packed and hashed as the older builds did.
    #[allow(clippy::too_many_arguments)]
    pub fn delivery(
        &self,
        id: &str,
        recipient_id: &str,
        sender: &str,
        event: &str,
        at: i64,
        text: &str,
        refs: &[&str],
        commit_seq: i64,
    ) {
        let body = zstd::bulk::compress(text.as_bytes(), 3).expect("compress");
        self.archive
            .execute(
                "INSERT INTO delivery(id, recipient_kind, recipient_id, sender, event, cwd, at,
                     content_hash, bytes, body, refs_json, left_out_json, commit_seq)
                 VALUES (?1, 'session', ?2, ?3, ?4, '/work/synthetic', ?5, ?6, ?7, ?8, ?9, '[]', ?10)",
                params![
                    id,
                    recipient_id,
                    sender,
                    event,
                    at,
                    blake3::hash(text.as_bytes()).as_bytes().as_slice(),
                    text.len() as i64,
                    body,
                    serde_json::to_string(refs).expect("refs"),
                    commit_seq
                ],
            )
            .expect("delivery");
    }

    /// The older `embeddings.db`, created on first use.
    pub fn embeddings(&mut self) -> &Connection {
        if self.embeddings.is_none() {
            let conn = Connection::open(self.root.join("embeddings.db")).expect("embeddings.db");
            conn.execute_batch(EMBEDDINGS_SQL)
                .expect("older embeddings declaration");
            conn.pragma_update(None, "user_version", 1)
                .expect("version");
            self.embeddings = Some(conn);
        }
        self.embeddings.as_ref().expect("embeddings")
    }

    /// Closes the databases, leaving the files as an older build would.
    pub fn finish(self) -> PathBuf {
        drop(self.embeddings);
        drop(self.archive);
        self.root
    }
}

/// One observation [`standard`] writes: ID, input, path, class, revision,
/// length, inode, details and seen time.
type ObservationFixture<'a> = (
    &'a str,
    &'a str,
    &'a str,
    &'a str,
    Option<i64>,
    Option<i64>,
    i64,
    Option<String>,
    i64,
);

/// What [`standard`] wrote, for tests to check against.
pub struct Standard {
    pub root: PathBuf,
    pub machine: String,
    /// The watched input with first, appended, rewritten, back and deleted observations.
    pub watched: String,
    /// The input whose file was deleted at its source after it was stored.
    pub deleted: String,
    /// The input imported from an export file, named by its bytes.
    pub imported: String,
    /// A blob no extent cites.
    pub loose_blob: BlobHash,
    /// The blob stored as a file.
    pub file_blob: BlobHash,
    pub vectors: u64,
}

pub const WATCHED_PATH: &str = "/synthetic/home/.agent/sessions/one.jsonl";
pub const DELETED_PATH: &str = "/synthetic/home/.agent/sessions/two.jsonl";
pub const IMPORTED_PATH: &str = "/synthetic/downloads/export.jsonl";

/// A synthetic older archive with a row of every class the importer carries:
/// a watched input through every observation class with inline and file
/// blobs (one shared by two inputs), an input deleted at its source, an
/// imported input, a blob no extent cites, notes with a revision, a tag and
/// a withdrawal, two values of one setting, a kept result, deliveries with
/// their events, and vectors with a permanent failure. Rows the importer
/// must refuse (a note kind with no counterpart, a delivery no request
/// wrote, a chat action) are included; [`STANDARD_REFUSALS`] names them.
pub fn standard(root: &Path) -> Standard {
    let mut b = Builder::create(root, 4);
    let machine = "0123456789abcdef0123456789abcdef";
    b.machine(machine, "workstation", 1_000);

    let first = b"{\"line\":1,\"text\":\"first record\"}\n".to_vec();
    let appended = b"{\"line\":2,\"text\":\"appended record\"}\n".to_vec();
    let rewritten = b"{\"line\":1,\"text\":\"rewritten record\"}\n".repeat(40);
    let a = b.inline_blob(&first);
    let file_blob = b.file_blob(&appended, 16);
    let c = b.inline_blob(&rewritten);
    let loose_blob = b.inline_blob(b"bytes no extent cites\n");

    let watched = "0190f000-0000-7000-8000-000000000001";
    let deleted = "0190f000-0000-7000-8000-000000000002";
    let export = b"{\"export\":true}\n".to_vec();
    let export_hash = b.inline_blob(&export);
    let imported = export_hash.to_string();
    let a_len = first.len() as i64;
    let b_len = appended.len() as i64;
    let c_len = rewritten.len() as i64;
    let sql = |statement: &str, values: &[&dyn rusqlite::ToSql]| {
        b.archive.execute(statement, values).expect(statement);
    };
    for (id, format) in [
        (watched, "claude-code"),
        (deleted, "codex"),
        (imported.as_str(), "claude-code"),
    ] {
        sql(
            "INSERT INTO input(id, format) VALUES (?1, ?2)",
            &[&id, &format],
        );
    }
    let revision = "INSERT INTO revision(input, number, produced, chain, length, commit_seq) VALUES (?1, ?2, ?3, ?4, ?5, ?6)";
    sql(revision, &[&watched, &1, &"first", &1, &a_len, &1]);
    sql(
        revision,
        &[&watched, &2, &"appended", &1, &(a_len + b_len), &2],
    );
    sql(revision, &[&watched, &3, &"rewritten", &3, &c_len, &3]);
    sql(revision, &[&deleted, &1, &"first", &1, &a_len, &4]);
    sql(
        revision,
        &[&imported, &1, &"imported", &1, &(export.len() as i64), &5],
    );
    let extent = "INSERT INTO extent(input, chain, offset, length, blob, blob_offset) VALUES (?1, ?2, ?3, ?4, ?5, ?6)";
    sql(extent, &[&watched, &1, &0, &a_len, &&a.0[..], &0]);
    sql(
        extent,
        &[&watched, &1, &a_len, &b_len, &&file_blob.0[..], &0],
    );
    sql(extent, &[&watched, &3, &0, &c_len, &&c.0[..], &0]);
    sql(extent, &[&deleted, &1, &0, &a_len, &&a.0[..], &0]);
    sql(
        extent,
        &[
            &imported,
            &1,
            &0,
            &(export.len() as i64),
            &&export_hash.0[..],
            &0,
        ],
    );
    let observation = "INSERT INTO observation(id, input, machine, path, class, revision, length,
        device, inode, size, mtime_ns, ctime_ns, details_json, polylogue_version, seen_at, commit_seq)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 64, ?8, ?7, ?9, ?9, ?10, '0.9.0', ?11, ?12)";
    let none: Option<String> = None;
    let no_revision: Option<i64> = None;
    let rows: [ObservationFixture; 9] = [
        (
            "o1",
            watched,
            WATCHED_PATH,
            "first",
            Some(1),
            Some(a_len),
            11,
            none.clone(),
            2_000,
        ),
        (
            "o2",
            watched,
            WATCHED_PATH,
            "appended",
            Some(2),
            Some(a_len + b_len),
            11,
            none.clone(),
            3_000,
        ),
        (
            "o3",
            watched,
            WATCHED_PATH,
            "rewritten",
            Some(3),
            Some(c_len),
            12,
            none.clone(),
            4_000,
        ),
        (
            "o4",
            watched,
            WATCHED_PATH,
            "deleted",
            no_revision,
            None,
            12,
            none.clone(),
            5_000,
        ),
        (
            "o5",
            watched,
            WATCHED_PATH,
            "back",
            Some(3),
            Some(c_len),
            12,
            none.clone(),
            6_000,
        ),
        (
            "o6",
            deleted,
            DELETED_PATH,
            "first",
            Some(1),
            Some(a_len),
            13,
            none.clone(),
            2_500,
        ),
        (
            "o7",
            deleted,
            DELETED_PATH,
            "deleted",
            no_revision,
            None,
            13,
            none.clone(),
            7_000,
        ),
        (
            "o8",
            imported.as_str(),
            IMPORTED_PATH,
            "imported",
            Some(1),
            Some(export.len() as i64),
            14,
            Some("{\"version\":1,\"import_id\":\"import-1\",\"member\":\"export.jsonl\"}".into()),
            8_000,
        ),
        (
            "o0",
            watched,
            WATCHED_PATH,
            "back",
            Some(3),
            Some(c_len),
            12,
            none,
            9_000,
        ),
    ];
    for (n, (id, input, path, class, rev, length, inode, details, seen)) in
        rows.into_iter().enumerate()
    {
        let mtime = seen * 1_000_000;
        sql(
            observation,
            &[
                &id,
                &input,
                &machine,
                &path,
                &class,
                &rev,
                &length,
                &inode,
                &mtime,
                &details,
                &seen,
                &(n as i64 + 1),
            ],
        );
    }

    // Notes: a note, its revision, a tag, a withdrawal of the tag, and a kind
    // with no counterpart here.
    let note = "INSERT INTO note(id, first_note, kind, key, value, author, machine, created_at, valid_until, commit_seq)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9)";
    let no_key: Option<&str> = None;
    sql(
        note,
        &[
            &"n1",
            &"n1",
            &"note",
            &no_key,
            &"a first note",
            &"operator",
            &machine,
            &10_000,
            &20,
        ],
    );
    sql(
        note,
        &[
            &"n2",
            &"n1",
            &"note",
            &no_key,
            &"the note revised",
            &"operator",
            &machine,
            &11_000,
            &21,
        ],
    );
    sql(
        note,
        &[
            &"t1",
            &"t1",
            &"tag",
            &Some("topic"),
            &"storage",
            &"operator",
            &machine,
            &12_000,
            &22,
        ],
    );
    sql(
        note,
        &[
            &"w1",
            &"w1",
            &"withdrawal",
            &no_key,
            &"no longer true",
            &"operator",
            &machine,
            &13_000,
            &23,
        ],
    );
    sql(
        note,
        &[
            &"x1",
            &"x1",
            &"distillate",
            &no_key,
            &"a kind with no counterpart",
            &"operator",
            &machine,
            &14_000,
            &24,
        ],
    );
    for (id, target) in [
        ("n1", "session:claude-code:one"),
        ("n2", "session:claude-code:one"),
        ("t1", "session:claude-code:one"),
        ("x1", "session:claude-code:one"),
    ] {
        sql(
            "INSERT INTO note_target(note_id, ref) VALUES (?1, ?2)",
            &[&id, &target],
        );
    }
    sql(
        "INSERT INTO note_relation(note_id, kind, other) VALUES ('n2', 'revises', 'note:n1')",
        &[],
    );
    sql(
        "INSERT INTO note_relation(note_id, kind, other) VALUES ('w1', 'withdraws', 'note:t1')",
        &[],
    );
    for (id, write, seq) in [
        ("n1", "add_note", 20),
        ("n2", "add_note", 21),
        ("t1", "add_note", 22),
        ("w1", "withdraw", 23),
    ] {
        b.request(id, 10_000, &serde_json::json!({"id": id, "machine": machine, "at": "1970-01-01T00:00:10Z", "write": {"write": write}}), seq);
    }

    // Two values of one archive-wide setting and one per-machine setting.
    let setting = "INSERT INTO setting(id, machine, key, value, author, set_at, commit_seq) VALUES (?1, ?2, ?3, ?4, 'operator', ?5, ?6)";
    sql(setting, &[&"s1", &"*", &"fetch-links", &"on", &15_000, &25]);
    sql(
        setting,
        &[&"s2", &"*", &"fetch-links", &"off", &16_000, &26],
    );
    sql(
        setting,
        &[
            &"s3",
            &machine,
            &"signal.interrupts_per_hour",
            &"6",
            &17_000,
            &27,
        ],
    );

    // A kept result and the request that kept it.
    sql(
        "INSERT INTO kept_result(id, name, query, ran_at, read_at, rows_json) VALUES ('k1', 'kept', 'sessions', 18000, '{\"at\":18000}', '{\"unit\":\"sessions\",\"columns\":[],\"rows\":[]}')",
        &[],
    );
    b.request("k1", 18_000, &serde_json::json!({"id": "k1", "machine": machine, "at": "1970-01-01T00:00:18Z", "write": {"write": "keep_result", "name": "kept", "query": "sessions", "rows_json": "{\"unit\":\"sessions\",\"columns\":[],\"rows\":[]}", "read_at": "{\"at\":18000}"}}), 28);

    // A prepared delivery with a delivered event, the event reported twice;
    // advice delivered by a hook; and a delivery no kept request wrote.
    let text = "synthetic context for a session";
    b.delivery(
        "d1",
        "claude-code:one",
        "hook",
        "session_start",
        19_000,
        text,
        &["session:claude-code:one"],
        29,
    );
    b.request("d1", 19_000, &serde_json::json!({"id": "d1", "machine": machine, "at": "1970-01-01T00:00:19Z", "write": {"write": "deliver", "recipient_kind": "session", "recipient_id": "claude-code:one", "sender": "hook", "event": "session_start", "cwd": "/work/synthetic", "text": text, "refs": ["session:claude-code:one"]}}), 29);
    sql(
        "INSERT INTO delivery_event(delivery, what, at, detail, commit_seq) VALUES ('d1', 'delivered', 19500, NULL, 30)",
        &[],
    );
    for (id, seq) in [("e1", 30), ("e2", 31)] {
        b.request(id, 19_500, &serde_json::json!({"id": id, "machine": machine, "at": "1970-01-01T00:00:19.500Z", "write": {"write": "delivery_event", "delivery": "d1", "what": "delivered"}}), seq);
    }
    let advice = "synthetic advice printed by a hook";
    b.delivery(
        "d2",
        "claude-code:one",
        "hook",
        "advice",
        20_000,
        advice,
        &[],
        32,
    );
    b.request("d2", 20_000, &serde_json::json!({"id": "d2", "machine": machine, "at": "1970-01-01T00:00:20Z", "write": {"write": "delivered_advice", "recipient_kind": "session", "recipient_id": "claude-code:one", "sender": "hook", "event": "advice", "cwd": "/work/synthetic", "text": advice, "refs": []}}), 32);
    sql(
        "INSERT INTO delivery_event(delivery, what, at, detail, commit_seq) VALUES ('d2', 'delivered', 20000, 'written to the harness hook output', 32)",
        &[],
    );
    b.delivery(
        "d3",
        "claude-code:one",
        "mcp",
        "recall",
        21_000,
        "a delivery without its request",
        &[],
        33,
    );

    // A queued chat action: an operational queue with no counterpart here.
    sql(
        "INSERT INTO chat_action(id, request_key, request_hash, request_json, stage_only, needs_approval, created_at, commit_seq)
         VALUES ('c1', 'key-1', X'00', '{}', 1, 0, 22000, 34)",
        &[],
    );

    sql(
        "INSERT INTO meta(key, value) VALUES ('write_counter', '34')",
        &[],
    );

    let embeddings = b.embeddings();
    let vector =
        |values: [f32; 2]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    embeddings
        .execute(
            "INSERT INTO embedded(text_hash, model, chunk, dims, vector, embedded_at) VALUES (?1, 'synthetic/model/2/document/unit', 0, 2, ?2, 23000)",
            params![&[1u8; 32][..], vector([0.6, 0.8])],
        )
        .expect("vector");
    embeddings
        .execute(
            "INSERT INTO embedded(text_hash, model, chunk, dims, vector, embedded_at) VALUES (?1, 'synthetic/model/2/document/unit', 1, 2, ?2, 23001)",
            params![&[1u8; 32][..], vector([1.0, 0.0])],
        )
        .expect("vector");
    embeddings
        .execute(
            "INSERT INTO embed_failure(text_hash, model, error, failed_at) VALUES (?1, 'synthetic/model/2/document/unit', 'rejected by the provider', 23002)",
            params![&[2u8; 32][..]],
        )
        .expect("failure");
    let root = b.finish();
    Standard {
        root,
        machine: machine.into(),
        watched: watched.into(),
        deleted: deleted.into(),
        imported,
        loose_blob,
        file_blob,
        vectors: 2,
    }
}

/// The rows [`standard`] holds that the importer refuses, as `(class, row)`.
pub const STANDARD_REFUSALS: [(&str, &str); 3] = [
    ("notes", "x1"),
    ("deliveries", "d3"),
    ("chat_action", "1 rows"),
];
