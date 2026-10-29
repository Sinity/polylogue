use super::*;

fn reading() -> crate::primary::Reading {
    crate::primary::Reading {
        input: crate::InputKey {
            id: "source".into(),
            format: "synthetic".into(),
            arrival_rank: 2,
        },
        chain: 1,
        reading_version: "v1".into(),
        order_from: None,
        repeats_whole_conversations: false,
        start: 0,
        parsed_upto: 2,
        at_end: false,
        no_parser_yet: false,
    }
}

fn staged(index: &mut Connection, product: Product) {
    stage(
        index,
        &[Confirmed {
            position: 1,
            output_index: 0,
            event: "native-fact".into(),
            change: ConfirmedChange::Live(Box::new(product)),
        }],
    )
    .unwrap();
    candidate(index, 1).unwrap();
}

#[test]
fn only_the_retained_control_supplies_a_parser_continuation() {
    let root = tempfile::tempdir().unwrap();
    let mut index = cache(&root.path().join("index.db"));
    let checkpoint = polylogue_model::batch::ParserCheckpoint::new(
        "v1".into(),
        2,
        Some(serde_json::json!({"native":"continuation"})),
    );
    staged(
        &mut index,
        Product::ParseProgress {
            reading: reading(),
            counts: Default::default(),
            reset: false,
            comparison: None,
            checkpoint: Some(checkpoint.clone()),
        },
    );
    let mut planned = plan(&index).unwrap();
    let mut actual = Vec::new();
    planned.readings[0]
        .3
        .visit_source(|step| {
            actual.push(step.checkpoint.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, vec![Some(checkpoint)]);
    drop(planned);
    let mut index = cache(&root.path().join("without-control.db"));
    staged(
        &mut index,
        Product::Session {
            reading: reading(),
            facts: polylogue_model::Session::new(
                polylogue_model::SessionId::parse("codex:session").unwrap(),
                polylogue_model::SessionKind::Main,
            ),
        },
    );
    let mut planned = plan(&index).unwrap();
    planned.readings[0]
        .3
        .visit_source(|step| {
            assert_eq!(step.checkpoint, None);
            assert!(step.source_prefix_missing);
            Ok(())
        })
        .unwrap();
}

#[test]
fn a_mismatched_canonical_continuation_is_refused_before_cache_application() {
    let root = tempfile::tempdir().unwrap();
    let mut index = cache(&root.path().join("index.db"));
    staged(
        &mut index,
        Product::ParseProgress {
            reading: reading(),
            counts: Default::default(),
            reset: false,
            comparison: None,
            checkpoint: Some(polylogue_model::batch::ParserCheckpoint::new(
                "other-version".into(),
                2,
                None,
            )),
        },
    );
    assert!(matches!(plan(&index), Err(StoreError::Invalid(_))));
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM parse_state", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}
