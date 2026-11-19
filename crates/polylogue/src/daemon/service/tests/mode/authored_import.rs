use super::*;
use polylogue_model::assertion::AssertionKind;
use polylogue_store::primary::{
    Capture, Confirmed, ConfirmedChange, Material, NamedProductSelector, Product, ReplayControl,
    Sinex as _,
};

fn control(control: ReplayControl) -> Confirmed {
    Confirmed {
        position: 0,
        event: String::new(),
        output_index: 0,
        sinex_context: None,
        change: ConfirmedChange::Control(Box::new(control)),
    }
}

fn restorations(connection: &polylogue_sinex::Sinex) -> Vec<Product> {
    let mut products = Vec::new();
    connection
        .visit_named_products(
            &NamedProductSelector::Prefix("assertion-restore:".into()),
            &mut |product| {
                products.push(product.clone());
                Ok(())
            },
        )
        .unwrap();
    products
}

fn confirm(writer: &polylogue_store::writer::WriterSender, position: i64, products: Vec<Product>) {
    let mut events: Vec<_> = products
        .into_iter()
        .enumerate()
        .map(|(ordinal, product)| Confirmed {
            position,
            event: format!("authored-import-{position}-{ordinal}"),
            output_index: ordinal as u64,
            sinex_context: None,
            change: ConfirmedChange::Live(Box::new(product)),
        })
        .collect();
    events.push(control(ReplayControl::Completion {
        position,
        delivery: None,
    }));
    writer.apply_confirmed(events).unwrap();
}

#[test]
fn python_authored_import_stays_pending_until_source_and_membership_are_confirmed() {
    let home = ClaudeHome::new();
    let source = home.dir.path().join("user.db");
    polylogue_store::python_polylogue::fixture::user_db(&source);
    {
        let db = rusqlite::Connection::open(&source).unwrap();
        db.execute(
            "DELETE FROM assertions WHERE assertion_id=?1",
            [polylogue_store::python_polylogue::fixture::MARK],
        )
        .unwrap();
        db.execute("UPDATE annotation_schemas SET registered_at_ms=0", [])
            .unwrap();
    }
    let original = std::fs::read(&source).unwrap();
    let mut startup = primary(&home);
    startup.engine.reads_sources = false;
    let runtime = launch(startup).unwrap();
    let writer = runtime.daemon.writes.as_ref().unwrap();
    let ModeState::Primary { connection } = &runtime.daemon.mode else {
        panic!("primary required")
    };
    writer
        .apply_confirmed(vec![control(ReplayControl::Reset {
            scope: "authored-import".into(),
        })])
        .unwrap();
    writer
        .apply_confirmed(vec![control(ReplayControl::Completion {
            position: 0,
            delivery: None,
        })])
        .unwrap();
    let import = || {
        crate::test_daemon::Call::new(
            "import",
            serde_json::json!({"paths":[source],"origin":"python-user-db"}),
        )
        .answer(&runtime.daemon)
    };
    let first = import();
    assert_eq!(
        first.exit_code(),
        5,
        "original facts must survive pending capture: {first:?}"
    );
    let report = first.value().unwrap();
    assert_eq!(report["python_import"]["pending"], true);
    assert_eq!(
        report["python_import"]["classes"]["assertions"]["pending"],
        11
    );
    assert_eq!(
        report["python_import"]["classes"]["annotation_schemas"]["pending"],
        1
    );
    let rows_input = report["python_import"]["rows_input"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        runtime
            .daemon
            .reads
            .reader()
            .unwrap()
            .input_progress(&rows_input)
            .unwrap()
            .is_none()
    );
    let facts = restorations(connection);
    assert_eq!(facts.len(), 13);
    assert!(
        facts
            .iter()
            .all(|product| matches!(product, Product::RestoredAssertion { import: None, .. }))
    );
    let rubric = facts
        .iter()
        .find_map(|product| match product {
            Product::RestoredAssertion { fact, .. } if fact.kind == AssertionKind::Rubric => {
                Some(fact.clone())
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(rubric.created_at, None);
    assert_eq!(
        runtime
            .daemon
            .reads
            .reader()
            .unwrap()
            .query("SELECT count(*) AS n FROM source_assertion", &[])
            .unwrap()[0]["n"],
        0
    );

    // Confirm the actual captured rows and the exact facts from the normal
    // producer, rather than making the importer believe capture was synchronous.
    let pending = connection.pending_captures().unwrap().into_iter().find(|entry|
        matches!(&entry.captured, Capture::Input(input) if input.rows.input == rows_input)).unwrap();
    let Capture::Input(captured) = pending.captured else {
        panic!("input capture")
    };
    let mut bodies = connection.capture_body(&pending.identity).unwrap();
    assert_eq!(bodies.len(), 1);
    let body = bodies.remove(0);
    let prepared = hashed_blobs::Prepared::from_encoded(
        body.file,
        body.descriptor,
        None,
        hashed_blobs::ReadLimits::default(),
    )
    .unwrap();
    let store = home.dir.path().join("sinex-store");
    let mut segment =
        hashed_blobs::SegmentWriter::open(&store, "segments/authored-source.hb", 0, &[], 0o600)
            .unwrap();
    let committed = segment
        .append_group([Ok(prepared)], |_| Ok(None))
        .unwrap()
        .wait()
        .unwrap();
    segment.acknowledge(&committed).unwrap();
    confirm(
        writer,
        1,
        vec![Product::Input {
            rows: Box::new(captured.rows.clone()),
            material: Some(Material {
                id: "authored-source".into(),
                hash: captured.hash,
                length: captured.length,
                extents: committed
                    .extents
                    .iter()
                    .map(hashed_blobs::Extent::encode)
                    .collect(),
            }),
        }],
    );
    confirm(writer, 2, facts.clone());
    let second = import();
    assert_eq!(
        second.exit_code(),
        5,
        "source membership still needs confirmation: {second:?}"
    );
    let all = restorations(connection);
    let membership: Vec<_> = all
        .iter()
        .filter(|product| {
            matches!(
                product,
                Product::RestoredAssertion {
                    import: Some(_),
                    ..
                }
            )
        })
        .cloned()
        .collect();
    assert_eq!(membership.len(), 1);
    let Product::RestoredAssertion {
        fact,
        import: Some(attribution),
    } = &membership[0]
    else {
        panic!("source membership")
    };
    assert_eq!(*fact, rubric);
    assert_eq!(attribution.input, rows_input);
    assert_eq!(
        attribution.chain,
        captured.rows.revision.as_ref().unwrap().chain
    );
    let before_pending_retry = serde_json::to_value(restorations(connection)).unwrap();
    let mut later_import = attribution.clone();
    later_import.at =
        polylogue_model::Timestamp::from_unix_ms(later_import.at.unix_ms() + 1_000).unwrap();
    let retry = writer
        .restore_assertion(rubric.clone(), Some(later_import))
        .unwrap();
    assert!(retry.pending);
    assert_eq!(
        serde_json::to_value(restorations(connection)).unwrap(),
        before_pending_retry,
        "a pending retry must retain its first observation time and exact publication"
    );
    confirm(writer, 3, membership);
    let before = serde_json::to_value(restorations(connection)).unwrap();
    let third = import();
    assert_eq!(
        third.exit_code(),
        0,
        "confirmed repeat must be complete: {third:?}"
    );
    assert_eq!(third.value().unwrap()["python_import"]["pending"], false);
    assert_eq!(
        serde_json::to_value(restorations(connection)).unwrap(),
        before,
        "strict repeat must add no publication"
    );
    let reader = runtime.daemon.reads.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM source_assertion", &[])
            .unwrap()[0]["n"],
        13
    );
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM source_assertion_import", &[])
            .unwrap()[0]["n"],
        1
    );
    assert_eq!(reader.query("SELECT count(*) AS n FROM v1_assertions WHERE kind='rubric' AND created_at IS NULL", &[]).unwrap()[0]["n"], 1);
    assert_eq!(std::fs::read(&source).unwrap(), original);
}
