//! Canonical metadata remains distinct from absent and derived clock evidence.
use super::*;
use polylogue_model::{Precision, TimeEvidence, TimeQuality};

fn at(text: &str) -> Timestamp {
    Timestamp::parse_rfc3339(text).unwrap()
}
fn query(reader: &crate::Reader<'_>, text: &str, fields: &[&str]) -> Vec<serde_json::Value> {
    query_result(reader, text, fields).unwrap_or_else(|error| panic!("{text}: {error}"))
}
fn query_result(
    reader: &crate::Reader<'_>,
    text: &str,
    fields: &[&str],
) -> crate::Result<Vec<serde_json::Value>> {
    let options = polylogue_query::Options {
        clock: query_language::time::Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::UTC,
                jiff::civil::Time::midnight(),
            )
            .unwrap(),
        ),
        origins: vec!["claude-code".into()],
        page_size: None,
        random_seed: Some(1),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        kept: Default::default(),
        fields: fields.iter().map(|field| (*field).into()).collect(),
        chat_pages: vec![],
    };
    let program =
        crate::query::compile(text, &options).unwrap_or_else(|error| panic!("{text}: {error}"));
    let parameters = program
        .params
        .iter()
        .map(|parameter| match parameter {
            crate::query::Param::Int(value) => crate::SqlValue::Int(*value),
            crate::query::Param::Real(value) => crate::SqlValue::Real(*value),
            crate::query::Param::Text(value) => crate::SqlValue::Text(value.clone()),
        })
        .collect::<Vec<_>>();
    reader.with_clock(&program.clock, || reader.query(&program.sql, &parameters))
}

#[test]
fn native_time_companions_use_evidence_and_keep_missing_relations_distinct() {
    let w = test_archive();
    let mut call = text_message(
        "metadata-call",
        "metadata",
        Role::Assistant,
        "synthetic native call",
        None,
    );
    call.set_time(
        Some(at("2026-10-29T10:00:00.123456789Z")),
        TimeEvidence {
            quality: TimeQuality::Operator,
            precision: Some(Precision::Bounded),
            until: Some(at("2026-10-29T11:00:00Z")),
        },
    )
    .unwrap();
    call.blocks.push(Block::ToolCall {
        call_id: Some("native-call".into()),
        tool: Some("shell".into()),
        action: polylogue_model::ToolAction::Shell,
        input: serde_json::json!({"command":"synthetic"}),
        command: Some("synthetic".into()),
        path: None,
        run_by_script: false,
    });
    let mut result = text_message("metadata-result", "metadata", Role::User, "", None);
    result
        .set_time(
            Some(at("2026-10-29T11:00:00Z")),
            TimeEvidence::intrinsic(Precision::Instant),
        )
        .unwrap();
    result.blocks = vec![Block::ToolResult {
        call_id: Some("native-call".into()),
        outcome: polylogue_model::ToolOutcome::Ok,
        output: Some("synthetic result".into()),
        data: None,
        attachments: vec![],
        reported_duration_ms: None,
    }];
    let mut input = batch(
        "time-companions",
        0,
        0,
        vec![row(1, call), row(2, result)],
        &["metadata", "child-unknown"],
    );
    input.parse[0].output.sessions[1].kind = SessionKind::Subagent;
    input.parse[0].output.sessions[0]
        .set_time(
            Some(at("2026-10-29T10:00:00Z")),
            TimeEvidence {
                quality: TimeQuality::Captured,
                precision: Some(Precision::Minute),
                until: None,
            },
        )
        .unwrap();
    for (index, (kind, time, evidence)) in [
        (
            "captured",
            Some(at("2026-10-29T10:30:00Z")),
            TimeEvidence {
                quality: TimeQuality::Captured,
                precision: Some(Precision::Instant),
                until: None,
            },
        ),
        (
            "calendar",
            Some(at("2026-01-01T00:00:00Z")),
            TimeEvidence {
                quality: TimeQuality::Intrinsic,
                precision: Some(Precision::Year),
                until: Some(at("2027-01-01T00:00:00Z")),
            },
        ),
        (
            "calendar-no-boundary",
            Some(at("2026-01-01T00:00:00Z")),
            TimeEvidence {
                quality: TimeQuality::Intrinsic,
                precision: Some(Precision::Year),
                until: None,
            },
        ),
        ("missing-time", None, TimeEvidence::UNKNOWN),
    ]
    .into_iter()
    .enumerate()
    {
        let mut event = Event {
            session: Some(sid("metadata")),
            kind: kind.into(),
            at: None,
            at_quality: TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            after: None,
            range: Some((index as u64 + 3, index as u64 + 4)),
            data: serde_json::json!({"synthetic":kind}),
        };
        event.set_time(time, evidence).unwrap();
        input.parse[0].output.events.push(event);
    }
    let mut document = polylogue_model::Document::new(
        polylogue_model::DocumentId::parse("claude-code:metadata-document").unwrap(),
        polylogue_model::DocumentPurpose::Context,
    );
    document.blocks.push(Block::Text {
        text: "synthetic document".into(),
    });
    document
        .set_time(
            Some(at("2026-10-29T10:00:00Z")),
            TimeEvidence {
                quality: TimeQuality::FileMtime,
                precision: Some(Precision::Second),
                until: None,
            },
        )
        .unwrap();
    input.parse[0]
        .output
        .documents
        .push(polylogue_model::DocumentRow {
            offset: 5,
            end: 6,
            document,
        });
    for child in ["missing-child", "child-unknown"] {
        input.parse[0].output.links.push(Link {
            kind: LinkKind::Spawn,
            parent: sid("metadata"),
            child: sid(child),
            at: None,
            call_id: None,
            evidence: "native-marker".into(),
        });
    }
    let bytes = serde_json::to_vec(&input.parse[0].output).unwrap();
    commit_event_bytes(&w, input, &bytes);
    w.writer.derive().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let message = query(
        &reader,
        "messages where id:claude-code:metadata-call",
        &[
            "id",
            "at.time_quality",
            "at.precision",
            "time_quality",
            "precision",
        ],
    );
    assert_eq!(message.len(), 1);
    assert_eq!(message[0]["at_time_quality"], "operator");
    assert_eq!(message[0]["time_quality"], "operator");
    assert_eq!(message[0]["at_precision"], "bounded");
    assert_eq!(message[0]["precision"], "bounded");
    let blocks = query(
        &reader,
        "blocks where message:claude-code:metadata-call",
        &["at.time_quality", "at.precision"],
    );
    assert_eq!(blocks.len(), 2);
    assert!(
        blocks
            .iter()
            .all(|row| row["at_time_quality"] == "operator" && row["at_precision"] == "bounded")
    );
    let calls = query(
        &reader,
        "tool_calls",
        &[
            "result_at",
            "at.time_quality",
            "at.precision",
            "result_at.time_quality",
            "result_at.precision",
        ],
    );
    assert_eq!(calls.len(), 1);
    assert!(calls[0]["result_at"].as_i64().is_some());
    assert_eq!(calls[0]["at_time_quality"], "operator");
    assert_eq!(calls[0]["at_precision"], "bounded");
    assert!(calls[0]["result_at_time_quality"].is_null());
    assert!(calls[0]["result_at_precision"].is_null());
    let sessions = query(
        &reader,
        "sessions where id:claude-code:metadata",
        &[
            "started_at.time_quality",
            "started_at.precision",
            "last_active.time_quality",
            "last_active.precision",
        ],
    );
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0]["started_at_time_quality"], "captured");
    assert_eq!(sessions[0]["started_at_precision"], "minute");
    // The completion is later than the call's whole bounded period.
    assert_eq!(sessions[0]["last_active_time_quality"], "intrinsic");
    assert_eq!(sessions[0]["last_active_precision"], "instant");
    let documents = query(
        &reader,
        "documents",
        &[
            "source_time.time_quality",
            "source_time.precision",
            "time_quality",
            "precision",
        ],
    );
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0]["source_time_time_quality"], "file_mtime");
    assert_eq!(documents[0]["time_quality"], "file_mtime");
    assert_eq!(documents[0]["source_time_precision"], "second");
    assert_eq!(documents[0]["precision"], "second");
    let calendar = query(
        &reader,
        "events where kind:calendar at:2026",
        &["at.precision"],
    );
    assert_eq!(calendar.len(), 1);
    assert_eq!(calendar[0]["at_precision"], "year");
    assert_eq!(
        query(
            &reader,
            "events where kind:calendar-no-boundary at.precision:year",
            &["kind"]
        )
        .len(),
        1
    );
    assert!(
        query(
            &reader,
            "events where kind:calendar-no-boundary at:2026",
            &["kind"]
        )
        .is_empty()
    );
    for (text, expected) in [
        ("events where at.time_quality:captured", vec!["captured"]),
        (
            "events where at.time_quality:\"unknown\"",
            vec!["missing-time"],
        ),
        ("events where at.time_quality:unknown", vec![]),
        ("events where at.precision:unknown", vec!["missing-time"]),
    ] {
        let rows = query(&reader, text, &["kind"]);
        assert_eq!(
            rows.iter()
                .map(|row| row["kind"].as_str().unwrap())
                .collect::<Vec<_>>(),
            expected,
            "{text}"
        );
    }
    for (text, expected) in [
        (
            "links where child.started_at.time_quality:unknown",
            vec!["claude-code:missing-child"],
        ),
        (
            "links where child.started_at.time_quality:\"unknown\"",
            vec!["claude-code:child-unknown"],
        ),
        (
            "links where NOT child.started_at.time_quality:intrinsic",
            vec!["claude-code:child-unknown"],
        ),
    ] {
        let rows = query(&reader, text, &["child"]);
        assert_eq!(
            rows.iter()
                .map(|row| row["child"].as_str().unwrap())
                .collect::<Vec<_>>(),
            expected,
            "{text}"
        );
    }
    let groups = query(
        &reader,
        "links | group by child.started_at.time_quality | count",
        &[],
    );
    assert_eq!(groups.len(), 2);
    assert!(
        groups
            .iter()
            .any(|row| row["child_started_at_time_quality"].is_null() && row["count"] == 1)
    );
    assert!(
        groups
            .iter()
            .any(|row| row["child_started_at_time_quality"] == "unknown" && row["count"] == 1)
    );
    // Metadata columns use canonical envelopes, independently of body decoders.
    drop(reader);
    drop(pool);
    let connection = crate::archive::open_write(&w.archive.index_db(), 64).unwrap();
    connection
        .execute_batch(
            "UPDATE message_version SET body=X'00'; UPDATE document_version SET body=X'00'; UPDATE document SET fields=X'00';",
        )
        .unwrap();
    let metadata_reads = ReadPool::open(&w.archive).unwrap();
    let reader = metadata_reads.reader().unwrap();
    assert_eq!(
        query(
            &reader,
            "messages where id:claude-code:metadata-call at.precision:bounded",
            &["id"]
        )
        .len(),
        1
    );
    assert_eq!(
        query(
            &reader,
            "documents where source_time.time_quality:file_mtime",
            &["id"]
        )
        .len(),
        1
    );
    drop(reader);
    drop(metadata_reads);
    connection
        .execute(
            "UPDATE message SET at_time=X'00' WHERE id='claude-code:metadata-call'",
            [],
        )
        .unwrap();
    let corrupt_reads = ReadPool::open(&w.archive).unwrap();
    let corrupt_reader = corrupt_reads.reader().unwrap();
    let failure = query_result(
        &corrupt_reader,
        "messages where id:claude-code:metadata-call at.time_quality:unknown",
        &["id"],
    )
    .unwrap_err();
    assert!(
        matches!(&failure, crate::StoreError::Sqlite(_)),
        "{failure:?}"
    );
}
