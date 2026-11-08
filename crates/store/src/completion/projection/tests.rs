use super::*;
use crate::completion::{CatalogReader, CatalogWriter, Metadata};
use crate::fixture::writer::{WriterArchive, prose};
use crate::writer::{ChainReading, InputKey, ParseStep};
use polylogue_model::{Output, RecordCounts, RecordOutcome};

fn remove_reading(archive: &WriterArchive, input: &str) {
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let progress = reads
        .reader()
        .unwrap()
        .input_progress(input)
        .unwrap()
        .unwrap();
    let mut counts = RecordCounts {
        records: 1,
        ..RecordCounts::default()
    };
    counts.add(
        "record",
        &RecordOutcome::skipped("replacement has no session facts"),
    );
    let mut step = ParseStep::new(Output::default(), counts, 0..progress.length, true);
    step.reading_version = "completion-removal=2".into();
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut step).unwrap();
    archive
        .writer
        .sender()
        .unwrap()
        .replace_chain(
            InputKey {
                id: input.into(),
                format: "claude-code".into(),
                arrival_rank: 0,
            },
            progress.chain,
            progress.length,
            reading,
        )
        .unwrap();
}

fn changes(
    reader: &Reader<'_>,
    after: &ChangePosition,
) -> (ChangePosition, Vec<String>, Vec<String>) {
    let mut upserted = Vec::new();
    let mut removed = Vec::new();
    let position = reader
        .visit_completion_sessions(Some(after), |change| {
            match change {
                SessionChange::Upsert(session) => upserted.push(session.id),
                SessionChange::Removed(id) => removed.push(id),
            }
            Ok(())
        })
        .unwrap();
    (position, upserted, removed)
}

#[test]
fn completion_removal_delta_accepts_an_empty_native_tombstone_feed() {
    let mut archive = WriterArchive::new();
    archive.write("retained", prose("retained", 1, 100, "first prompt"));
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    let before = reader.change_position().unwrap();
    let (position, upserted, removed) = changes(&reader, &before);
    assert_eq!(position, before);
    assert!(upserted.is_empty());
    assert!(removed.is_empty());
}

#[test]
fn completion_removal_delta_follows_real_writer_reading_replacement() {
    let mut archive = WriterArchive::new();
    archive.write("removed", prose("removed", 1, 100, "original prompt"));
    archive.write("survivor", prose("survivor", 1, 100, "surviving prompt"));
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let before = reads.reader().unwrap().change_position().unwrap();
    remove_reading(&archive, "input-removed");
    let reader = reads.reader().unwrap();
    let (position, _, removed) = changes(&reader, &before);
    assert!(position.index > before.index);
    assert_eq!(removed, ["claude-code:removed"]);
    let tombstone = reader
        .query("SELECT session_id,changed FROM removed_session", &[])
        .unwrap();
    assert_eq!(tombstone.len(), 1);
    assert_eq!(tombstone[0]["session_id"], "claude-code:removed");
    let removed_at = tombstone[0]["changed"].as_i64().unwrap();
    assert!(removed_at > before.index && removed_at <= position.index);
    let mut current = Vec::new();
    reader
        .visit_completion_sessions(None, |change| {
            if let SessionChange::Upsert(session) = change {
                current.push(session.id);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(current, ["claude-code:survivor"]);
    let (_, _, removed_again) = changes(&reader, &position);
    assert!(removed_again.is_empty());
}

#[test]
fn completion_removal_delta_keeps_a_recreated_session_in_the_same_cursor_window() {
    let mut archive = WriterArchive::new();
    archive.write("restored", prose("restored", 1, 100, "old prompt"));
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let initial = reads.reader().unwrap();
    let before = initial.change_position().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let metadata = Metadata {
        archive: "synthetic".into(),
        position: before.clone(),
        written_at: 1,
    };
    let mut catalog = CatalogWriter::build(&path, &metadata, |writer| {
        initial.visit_completion_sessions(None, |change| {
            if let SessionChange::Upsert(session) = change {
                writer.put_session(&session)?;
            }
            Ok(())
        })?;
        Ok(())
    })
    .unwrap();
    drop(initial);
    remove_reading(&archive, "input-restored");
    archive.write("restored", prose("restored", 2, 200, "new prompt"));
    let reader = reads.reader().unwrap();
    let (position, upserted, removed) = changes(&reader, &before);
    assert!(upserted.iter().any(|id| id == "claude-code:restored"));
    assert!(removed.is_empty());
    catalog
        .update(
            &Metadata {
                position,
                written_at: 2,
                ..metadata
            },
            |writer| {
                reader.visit_completion_sessions(Some(&before), |change| match change {
                    SessionChange::Upsert(session) => writer.put_session(&session),
                    SessionChange::Removed(id) => writer.remove_session(&id),
                })?;
                Ok(())
            },
        )
        .unwrap();
    let cache = CatalogReader::open(&path, "synthetic").unwrap().unwrap();
    assert_eq!(
        cache
            .session_details("claude-code:restored")
            .unwrap()
            .unwrap()
            .first_prompt
            .as_deref(),
        Some("new prompt")
    );
    // The reader excludes stale tombstones using the same live-session snapshot
    // that supplies upserts; the caller need not invent an event order.
    assert_eq!(
        reader
            .query("SELECT session_id FROM removed_session", &[])
            .unwrap(),
        vec![serde_json::json!({"session_id":"claude-code:restored"})]
    );
}

#[test]
fn authored_deltas_follow_only_affected_families_and_setting_keys() {
    use polylogue_model::{Reference, user::UserWrite};
    let archive = WriterArchive::new();
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let write = |id: &str, value| {
        archive
            .writer
            .user_preparing(
                crate::fixture::authored::operator_request(id.into(), value),
                crate::user_writes::PreparationContext::new(vec!["codex".into()]),
                None,
            )
            .unwrap()
    };
    for (id, name) in [("saved-a", "alpha"), ("saved-b", "beta")] {
        write(
            id,
            UserWrite::SaveQuery {
                name: name.into(),
                text: "sessions | group by origin | count".into(),
                denominator: None,
            },
        );
    }
    write(
        "setting-a",
        UserWrite::SetSetting {
            key: "signal.interrupts_per_hour".into(),
            value: "3".into(),
            machine: None,
        },
    );
    let after = reads.reader().unwrap().change_position().unwrap();
    write(
        "setting-b",
        UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
    );
    {
        let reader = reads.reader().unwrap();
        let mut definitions = Vec::new();
        reader
            .visit_completion_definitions(Some(after.archive), |value| {
                definitions.push(value.family);
                Ok(())
            })
            .unwrap();
        assert!(
            definitions.is_empty(),
            "an unrelated setting must not reload saved families: {definitions:?}"
        );
        assert_eq!(
            reader.completion_setting_keys(Some(after.archive)).unwrap(),
            ["browser.actions"]
        );
    }
    let after = reads.reader().unwrap().change_position().unwrap();
    write(
        "withdraw-a",
        UserWrite::WithdrawAssertions {
            targets: vec![Reference::parse("assertion:saved-a").unwrap()],
            reason: None,
        },
    );
    let mut changes = Vec::new();
    reads
        .reader()
        .unwrap()
        .visit_completion_definitions(Some(after.archive), |value| {
            changes.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].family, "saved-a");
    assert!(changes[0].saved.is_none() && changes[0].queries.is_empty());
    let after = reads.reader().unwrap().change_position().unwrap();
    write(
        "reinstate-a",
        UserWrite::WithdrawAssertions {
            targets: vec![Reference::parse("assertion:withdraw-a").unwrap()],
            reason: None,
        },
    );
    let mut changes = Vec::new();
    reads
        .reader()
        .unwrap()
        .visit_completion_definitions(Some(after.archive), |value| {
            changes.push(value);
            Ok(())
        })
        .unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].family, "saved-a");
    assert_eq!(changes[0].saved.as_ref().unwrap().name, "alpha");
    assert!(changes[0].queries.contains_key("alpha"));
    let after = reads.reader().unwrap().change_position().unwrap();
    write(
        "replace-a",
        UserWrite::SaveQuery {
            name: "alpha".into(),
            text: "messages | group by role | count".into(),
            denominator: None,
        },
    );
    let delta = reads
        .reader()
        .unwrap()
        .completion_vocabulary_changes(Some(&after))
        .unwrap();
    assert!(!delta.replaced);
    assert_eq!(delta.definitions.len(), 1);
    assert_eq!(delta.definitions[0].family, "saved-a");
    assert_eq!(
        delta.definitions[0].saved.as_ref().unwrap().text,
        "messages | group by role | count"
    );
}
