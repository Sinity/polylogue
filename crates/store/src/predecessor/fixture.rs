//! A synthetic predecessor `user.db`, declared with the predecessor's own
//! table statements, holding one row of every form the import maps and one
//! it refuses.

use rusqlite::{Connection, params};
use std::path::Path;

/// The predecessor's declarations of the tables these rows use.
pub const USER_DB_SQL: &str = r#"
CREATE TABLE assertions (
    assertion_id        TEXT PRIMARY KEY,
    scope_ref           TEXT,
    target_ref          TEXT NOT NULL,
    key                 TEXT,
    kind                TEXT NOT NULL,
    value_json          TEXT,
    body_text           TEXT,
    author_ref          TEXT DEFAULT 'user:local',
    author_kind         TEXT DEFAULT 'user',
    evidence_refs_json  TEXT DEFAULT '[]',
    status              TEXT DEFAULT 'active',
    visibility          TEXT DEFAULT 'private',
    confidence          REAL,
    staleness_json      TEXT,
    context_policy_json TEXT DEFAULT '{"inject":false}',
    supersedes_json     TEXT DEFAULT '[]',
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL
) STRICT;
CREATE TABLE annotation_schemas (
    schema_id          TEXT NOT NULL,
    schema_version     INTEGER NOT NULL,
    definition_json    TEXT NOT NULL,
    definition_sha256  TEXT NOT NULL,
    registered_at_ms   INTEGER NOT NULL,
    PRIMARY KEY (schema_id, schema_version)
) STRICT;
CREATE TABLE result_set_members (
    result_set_id TEXT NOT NULL,
    rank          INTEGER NOT NULL CHECK(rank >= 0),
    member_ref    TEXT NOT NULL CHECK(length(trim(member_ref)) > 0),
    PRIMARY KEY (result_set_id, rank),
    UNIQUE (result_set_id, member_ref)
) STRICT;
"#;

pub const DETECTOR: &str = "insight:duplicate-capture@v1";
pub const SESSION: &str = "session:chatgpt-export:00000000-0000-4000-8000-00000000aaaa";
pub const CANDIDATE: &str = "judgment:candidate-open";
pub const ACCEPTED: &str = "judgment:candidate-accepted";
pub const REJECTED: &str = "judgment:candidate-rejected";
pub const PROMOTED: &str = "assertion-promoted:accepted";
pub const ACCEPT: &str = "assertion-judgment:accept";
pub const REJECT: &str = "assertion-judgment:reject";
pub const BLOCKER: &str = "blocker:competing-copy";
pub const FINDING: &str = "assertion-finding:followups";
pub const NOTE: &str = "assertion-note:handoff";
pub const MARK: &str = "mark:star";
pub const AGENT_SESSION: &str = "codex-session:00000000-0000-7000-8000-00000000c0de";
pub const RESULT_SET: &str = "finding-0001";
pub const MEMBERS: [&str; 2] = [
    "message:claude-code-session:00000000-0000-4000-8000-00000000bbbb:m-1",
    "message:claude-code-session:00000000-0000-4000-8000-00000000bbbb:m-2",
];

/// Writes the synthetic `user.db` at `path`: twelve assertions (eleven
/// mapped, one of a kind with no counterpart), one annotation schema and a
/// result set of two members.
pub fn user_db(path: &Path) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(USER_DB_SQL).unwrap();
    let insert = |id: &str,
                  target: &str,
                  key: Option<&str>,
                  kind: &str,
                  value: Option<&str>,
                  body: &str,
                  author: (&str, &str),
                  evidence: &str,
                  status: &str,
                  staleness: Option<&str>,
                  supersedes: &str,
                  at: i64| {
        conn.execute(
            "INSERT INTO assertions(assertion_id, scope_ref, target_ref, key, kind, value_json, body_text,
                author_ref, author_kind, evidence_refs_json, status, staleness_json, supersedes_json,
                created_at_ms, updated_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?14)",
            params![
                id,
                DETECTOR,
                target,
                key,
                kind,
                value,
                body,
                author.0,
                author.1,
                evidence,
                status,
                staleness,
                supersedes,
                at
            ],
        )
        .unwrap();
    };
    let detector = (DETECTOR, "detector");
    let user = ("user:morrow", "user");
    let request = r#"{"actuator":"request_judgment","state":"needs_judgment"}"#;
    insert(
        CANDIDATE,
        SESSION,
        Some("duplicate:1"),
        "judgment",
        Some(request),
        "two captures of one conversation disagree",
        detector,
        "[]",
        "candidate",
        None,
        "[]",
        1_000,
    );
    insert(
        ACCEPTED,
        SESSION,
        Some("duplicate:2"),
        "judgment",
        Some(request),
        "a recapture repeats the stored copy",
        detector,
        "[]",
        "accepted",
        None,
        "[]",
        1_001,
    );
    insert(
        REJECTED,
        SESSION,
        Some("duplicate:3"),
        "judgment",
        Some(request),
        "an older plan names a copy that is gone",
        detector,
        "[]",
        "rejected",
        None,
        "[]",
        1_002,
    );
    insert(
        PROMOTED,
        SESSION,
        Some("duplicate:2"),
        "judgment",
        Some(request),
        "a recapture repeats the stored copy",
        user,
        &format!(r#"["assertion:{ACCEPTED}"]"#),
        "active",
        None,
        &format!(r#"["assertion:{ACCEPTED}"]"#),
        2_000,
    );
    insert(
        ACCEPT,
        &format!("assertion:{ACCEPTED}"),
        Some("accept/1"),
        "judgment",
        Some(&format!(
            r#"{{"decision":"accept","resulting_assertion_ref":"assertion:{PROMOTED}"}}"#
        )),
        "the bytes are identical",
        user,
        &format!(r#"["assertion:{ACCEPTED}","assertion:{PROMOTED}"]"#),
        "active",
        None,
        &format!(r#"["assertion:{ACCEPTED}"]"#),
        2_000,
    );
    insert(
        REJECT,
        &format!("assertion:{REJECTED}"),
        Some("reject/1"),
        "judgment",
        Some(r#"{"decision":"reject","resulting_assertion_ref":null}"#),
        "the plan it names no longer exists",
        user,
        &format!(r#"["assertion:{REJECTED}"]"#),
        "active",
        None,
        &format!(r#"["assertion:{REJECTED}"]"#),
        2_001,
    );
    insert(
        BLOCKER,
        SESSION,
        Some("raw/1"),
        "blocker",
        Some(r#"{"reason":"competing copy"}"#),
        "a competing copy blocks the merge",
        detector,
        "[]",
        "candidate",
        None,
        "[]",
        3_000,
    );
    insert(
        FINDING,
        "analysis:followups",
        Some("finding.followups"),
        "finding",
        Some(&format!(
            r#"{{"n":2,"result_set_ref":"result-set:{RESULT_SET}"}}"#
        )),
        "both failures were followed by a retry",
        ("run:followups-1", "detector"),
        r#"["query:00","result-set:finding-0001"]"#,
        "candidate",
        None,
        "[]",
        4_000,
    );
    insert(
        NOTE,
        "session:claude-code-session:00000000-0000-4000-8000-00000000bbbb",
        Some("thread-1"),
        "note",
        None,
        "pick up the open question tomorrow",
        (&format!("agent:{AGENT_SESSION}"), "agent"),
        &format!(r#"["{AGENT_SESSION}","path:notes/today.md"]"#),
        "candidate",
        Some(r#"{"expires_at":"2026-07-12T00:00:00+00:00"}"#),
        "[]",
        5_000,
    );
    insert(
        MARK,
        SESSION,
        Some("star"),
        "mark",
        None,
        "starred",
        user,
        "[]",
        "active",
        None,
        "[]",
        6_000,
    );
    // Two more open requests, so every form is not a singleton.
    insert(
        "judgment:candidate-open-2",
        SESSION,
        Some("duplicate:4"),
        "judgment",
        Some(request),
        "a third capture disagrees",
        detector,
        "[]",
        "candidate",
        None,
        "[]",
        1_003,
    );
    insert(
        "judgment:candidate-open-3",
        SESSION,
        Some("duplicate:5"),
        "judgment",
        Some(request),
        "a fourth capture disagrees",
        detector,
        "[]",
        "candidate",
        None,
        "[]",
        1_004,
    );
    conn.execute(
        "INSERT INTO annotation_schemas VALUES ('seed.example', 1, '{\"schema_id\":\"seed.example\",\"version\":1}', ?1, 0)",
        ["0".repeat(64)],
    )
    .unwrap();
    for (rank, member) in MEMBERS.iter().enumerate() {
        conn.execute(
            "INSERT INTO result_set_members VALUES (?1, ?2, ?3)",
            params![RESULT_SET, rank as i64, member],
        )
        .unwrap();
    }
}

/// The backing tables sqlite-vec declares for `message_embeddings`, as the
/// predecessor's stores hold them, and each layout's metadata table.
pub const VECTOR_BACKING_SQL: &str = r#"
CREATE TABLE "message_embeddings_info" (key text primary key, value any);
CREATE TABLE "message_embeddings_chunks"(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,size INTEGER NOT NULL,validity BLOB NOT NULL,rowids BLOB NOT NULL);
CREATE TABLE "message_embeddings_rowids"(rowid INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,chunk_id INTEGER,chunk_offset INTEGER);
CREATE TABLE "message_embeddings_vector_chunks00"(rowid PRIMARY KEY,vectors BLOB NOT NULL);
"#;
pub const ADDRESSED_META_SQL: &str = r#"
CREATE TABLE message_embeddings_meta (
    embedding_input_hash   BLOB PRIMARY KEY CHECK(length(embedding_input_hash) = 32),
    model                  TEXT NOT NULL,
    dimension              INTEGER NOT NULL,
    embedded_at_ms         INTEGER,
    recipe_hash            BLOB CHECK(recipe_hash IS NULL OR length(recipe_hash) = 32),
    output_contract_hash   BLOB CHECK(output_contract_hash IS NULL OR length(output_contract_hash) = 32)
) STRICT;
"#;
pub const BY_MESSAGE_META_SQL: &str = r#"
CREATE TABLE message_embeddings_meta (
    message_id      TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    dimension       INTEGER NOT NULL,
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32),
    embedded_at_ms  INTEGER,
    needs_reindex   INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0, 1))
) STRICT;
"#;

/// Vectors per backing chunk in these fixtures; sqlite-vec uses 1024.
pub const CHUNK: usize = 2;

/// One vector of a fixture store: its key (an address, or a message ID) and values.
pub enum VectorKey {
    Address([u8; 32]),
    Message(String),
}

/// Writes a vector store at `path` holding `vectors` of model `model`,
/// `CHUNK` to a backing chunk, with one empty slot left unset in the last
/// chunk when the count is odd. Every vector is embedded at `at`.
pub fn vector_store(path: &Path, model: &str, vectors: &[(VectorKey, Vec<f32>)], at: i64) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(VECTOR_BACKING_SQL).unwrap();
    let addressed = matches!(vectors.first(), Some((VectorKey::Address(_), _)));
    conn.execute_batch(if addressed {
        ADDRESSED_META_SQL
    } else {
        BY_MESSAGE_META_SQL
    })
    .unwrap();
    let dims = vectors.first().map_or(0, |(_, values)| values.len());
    for (chunk_index, chunk) in vectors.chunks(CHUNK).enumerate() {
        let chunk_id = chunk_index as i64 + 1;
        let mut validity = vec![0u8; CHUNK.div_ceil(8)];
        let mut bytes = vec![0u8; CHUNK * dims * 4];
        for (offset, (key, values)) in chunk.iter().enumerate() {
            validity[offset / 8] |= 1 << (offset % 8);
            for (i, value) in values.iter().enumerate() {
                let at = (offset * dims + i) * 4;
                bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
            let id = match key {
                VectorKey::Address(address) => {
                    let hex: String = address.iter().map(|b| format!("{b:02x}")).collect();
                    conn.execute(
                        "INSERT INTO message_embeddings_meta(embedding_input_hash, model, dimension, embedded_at_ms) VALUES (?1, ?2, ?3, ?4)",
                        params![address.to_vec(), model, dims as i64, at],
                    )
                    .unwrap();
                    hex
                }
                VectorKey::Message(id) => {
                    conn.execute(
                        "INSERT INTO message_embeddings_meta(message_id, model, dimension, content_hash, embedded_at_ms) VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![id, model, dims as i64, vec![0u8; 32], at],
                    )
                    .unwrap();
                    id.clone()
                }
            };
            conn.execute(
                "INSERT INTO message_embeddings_rowids(id, chunk_id, chunk_offset) VALUES (?1, ?2, ?3)",
                params![id, chunk_id, offset as i64],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO message_embeddings_chunks(chunk_id, size, validity, rowids) VALUES (?1, ?2, ?3, ?4)",
            params![chunk_id, CHUNK as i64, validity, vec![0u8; CHUNK * 8]],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO message_embeddings_vector_chunks00(rowid, vectors) VALUES (?1, ?2)",
            params![chunk_id, bytes],
        )
        .unwrap();
    }
}
