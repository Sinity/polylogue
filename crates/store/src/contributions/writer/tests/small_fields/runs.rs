use super::*;
use crate::contributions::{SmallFieldEntry, SourceRange, SourceRangeEntry};
fn fields() -> Vec<SmallFieldEntry> {
    ["one", "two", "three"]
        .into_iter()
        .enumerate()
        .map(|(i, key)| SmallFieldEntry {
            slot: slot(key),
            source_time: Some(7),
            supplied: true,
            block: false,
            field: payload(),
            expected_sequence: i as u64,
            next_cursor: key.as_bytes().to_vec(),
        })
        .collect()
}
fn ranges() -> Vec<SourceRangeEntry> {
    [(0, 3), (3, 7), (0, 3)]
        .into_iter()
        .enumerate()
        .map(|(i, (start, end))| SourceRangeEntry {
            range: SourceRange { start, end },
            expected_sequence: i as u64,
            next_cursor: vec![i as u8],
        })
        .collect()
}
fn parsed(f: &mut Fixture) {
    f.writer
        ._index
        .execute_batch(
            "UPDATE parse_state SET parsed_upto=acquired_upto,inspected_upto=acquired_upto",
        )
        .unwrap();
}
fn signature(f: &Fixture) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    f.writer._index.prepare("SELECT s.semantic_key,v.hash,o.hash FROM message_field_fact f JOIN message_field_slot s ON s.pk=f.slot_pk JOIN body_value v ON v.pk=f.value_pk JOIN body_value o ON o.pk=f.origin_value_pk ORDER BY s.semantic_key").unwrap().query_map([],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap().collect::<std::result::Result<_,_>>().unwrap()
}
#[test]
fn typed_field_run_matches_single_field_facts_roots_and_every_sequence_after_reopen() {
    let mut a = Fixture::new();
    let ba = a.prepare().unwrap().build;
    let mut b = Fixture::new();
    let bb = b.prepare().unwrap().build;
    a.writer
        .write_small_field_run(&ba, &fields(), 65536)
        .unwrap();
    for e in fields() {
        b.writer
            .write_small_field(
                &bb,
                &e.slot,
                e.source_time,
                e.supplied,
                e.block,
                &e.field,
                &ProjectionStep {
                    expected_sequence: e.expected_sequence,
                    next_cursor: &e.next_cursor,
                },
                65536,
            )
            .unwrap();
    }
    let a = a.reopen();
    let b = b.reopen();
    assert_eq!(signature(&a), signature(&b));
    assert_eq!(
        a.writer.message_projection(&ba, 65536).unwrap(),
        b.writer.message_projection(&bb, 65536).unwrap()
    );
    assert_eq!(a.scalar("SELECT count(*) FROM message_field_fact"), 3);
    assert_eq!(
        a.scalar("SELECT count(*) FROM body_value WHERE state='ready'"),
        1
    );
}
#[test]
fn typed_runs_late_member_failure_rolls_back_values_descriptors_fields_ranges_and_sequence() {
    for citations in [false, true] {
        let mut f = Fixture::new();
        parsed(&mut f);
        let build = f.prepare().unwrap().build;
        if citations {
            let mut entries = ranges();
            entries[2].expected_sequence = 1;
            assert!(matches!(
                f.writer.add_message_source_run(&build, &entries, 65536),
                Err(Error::StaleMessage)
            ));
        } else {
            let mut entries = fields();
            entries[2].expected_sequence = 1;
            assert!(matches!(
                f.writer.write_small_field_run(&build, &entries, 65536),
                Err(Error::StaleMessage)
            ));
        }
        let f = f.reopen();
        for table in [
            "body_value",
            "body_value_chunk",
            "message_field_fact",
            "message_source",
        ] {
            assert_eq!(f.scalar(&format!("SELECT count(*) FROM {table}")), 0);
        }
        assert_eq!(
            f.writer.message_projection(&build, 65536).unwrap().sequence,
            0
        );
    }
}
#[test]
fn typed_runs_keep_live_hidden_removal_contributor_excision_and_source_guards() {
    for sql in [
        "UPDATE message SET hidden=1",
        "UPDATE message SET removal_epoch=removal_epoch+1",
        "UPDATE message SET contributor_revision=contributor_revision+1",
        "UPDATE input_ref SET excised=1",
        "UPDATE parse_state SET state_generation=state_generation+1",
    ] {
        for citations in [false, true] {
            let mut f = Fixture::new();
            parsed(&mut f);
            let build = f.prepare().unwrap().build;
            f.writer._index.execute_batch(sql).unwrap();
            let refused = if citations {
                f.writer.add_message_source_run(&build, &ranges(), 65536)
            } else {
                f.writer.write_small_field_run(&build, &fields(), 65536)
            };
            assert!(refused.is_err(), "{sql}");
            for table in ["body_value", "message_field_fact", "message_source"] {
                assert_eq!(f.scalar(&format!("SELECT count(*) FROM {table}")), 0);
            }
            assert_eq!(f.scalar("SELECT source_cursor FROM message_build"), 0);
        }
    }
}
#[test]
fn aggregate_admission_and_typed_key_failure_happen_without_partial_install() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let entries = fields();
    let one = entries[0].retained_bytes(65536).unwrap();
    assert!(matches!(
        f.writer.write_small_field_run(&build, &entries, one),
        Err(Error::ParserAdmission { .. })
    ));
    let mut bad = fields();
    bad[1].slot = slot(&"x".repeat(2048));
    assert!(matches!(
        bad[1].retained_bytes(1024),
        Err(Error::ParserAdmission { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
    assert_eq!(
        f.writer.message_projection(&build, 65536).unwrap().sequence,
        0
    );
}
#[test]
fn source_range_run_matches_single_source_set_and_duplicate_semantics() {
    let mut a = Fixture::new();
    parsed(&mut a);
    let ba = a.prepare().unwrap().build;
    let mut b = Fixture::new();
    parsed(&mut b);
    let bb = b.prepare().unwrap().build;
    a.writer
        .add_message_source_run(&ba, &ranges(), 65536)
        .unwrap();
    for e in ranges() {
        b.writer
            .add_message_source(
                &bb,
                e.range,
                &ProjectionStep {
                    expected_sequence: e.expected_sequence,
                    next_cursor: &e.next_cursor,
                },
                65536,
            )
            .unwrap();
    }
    assert_eq!(a.scalar("SELECT count(*) FROM message_source"), 2);
    assert_eq!(
        a.writer.message_projection(&ba, 65536).unwrap(),
        b.writer.message_projection(&bb, 65536).unwrap()
    );
    assert_eq!(
        a.scalar("SELECT sum(source_start+source_end) FROM message_source"),
        b.scalar("SELECT sum(source_start+source_end) FROM message_source")
    );
}
#[test]
fn scoped_logical_reads_observe_first_action_then_bound_all_adapters_and_restore_on_panic() {
    let mut f = Fixture::new();
    let _ = f.prepare().unwrap();
    let snapshot = f.writer.parser_snapshot(&f.head.token).unwrap();
    let scope = snapshot
        .bounded_reads(parser_state::Budget {
            cells: 2,
            encoded_bytes: 1000,
        })
        .unwrap();
    assert!(snapshot.cell("absent", b"a", 100).unwrap().is_none());
    assert!(scope.enforce());
    assert!(
        snapshot
            .neighbor("absent", b"b", true, 100)
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        snapshot.cell("absent", b"c", 100),
        Err(Error::ParserAdmission {
            resource: "worker read cells",
            ..
        })
    ));
    drop(scope);
    let source_scope = snapshot
        .bounded_reads(parser_state::Budget {
            cells: 1,
            encoded_bytes: 10,
        })
        .unwrap();
    snapshot.charge_selector_record(10).unwrap();
    assert!(source_scope.enforce());
    assert!(matches!(
        snapshot.charge_selector_record(1),
        Err(Error::ParserAdmission {
            resource: "worker read cells",
            ..
        })
    ));
    drop(source_scope);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _scope = snapshot
            .bounded_reads(parser_state::Budget {
                cells: 0,
                encoded_bytes: 0,
            })
            .unwrap();
        panic!("synthetic worker panic");
    }));
    assert!(snapshot.cell("absent", b"d", 100).unwrap().is_none());
    let scope = snapshot
        .bounded_reads(parser_state::Budget {
            cells: 1,
            encoded_bytes: 2,
        })
        .unwrap();
    assert!(snapshot.cell("absent", b"d", 100).unwrap().is_none());
    assert!(!scope.enforce());
}

#[test]
fn every_member_rechecks_live_authority_and_rolls_back_triggered_removal() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.writer._index.execute_batch("CREATE TRIGGER synthetic_hide AFTER INSERT ON message_field_fact BEGIN UPDATE message SET hidden=1; END;").unwrap();
    assert!(matches!(
        f.writer.write_small_field_run(&build, &fields(), 65536),
        Err(Error::StaleMessage)
    ));
    assert_eq!(f.scalar("SELECT hidden FROM message"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
    assert_eq!(f.scalar("SELECT source_cursor FROM message_build"), 0);
}
