use super::*;
use polylogue_model::batch::{ExportDelivery, ExportDeliverySource};
use polylogue_store::primary::ExportReadingKey;

fn key(length: u64) -> ExportReadingKey {
    ExportReadingKey {
        input: polylogue_store::InputKey {
            id: "input-x".into(),
            format: "codex".into(),
            route: polylogue_model::Route::Live,
        },
        chain: 1,
        reading_version: "export-2".into(),
        length,
    }
}

fn callback(
    key: &ExportReadingKey,
    ordinal: u64,
    previous: BlobHash,
    complete: bool,
) -> Vec<ProductWrite> {
    let mut output = Output::default();
    if ordinal == 1 {
        output.sessions.push(polylogue_model::Session::new(
            polylogue_model::SessionId::parse("codex:export-session").unwrap(),
            polylogue_model::SessionKind::Main,
        ));
    }
    let mut step = polylogue_store::ParseStep::new(
        output,
        RecordCounts::default(),
        0..if complete { key.length } else { 0 },
        complete,
    );
    step.reading_version = key.reading_version.clone();
    let source = if complete {
        ExportDeliverySource::Complete {
            length: key.length,
            item_count: 1,
            delivery_count: ordinal,
        }
    } else if ordinal == 0 {
        ExportDeliverySource::ContainerHeader
    } else {
        ExportDeliverySource::Item { start: 4, end: 10 }
    };
    step.seal_export(ordinal, source, previous).unwrap();
    step.prepare();
    polylogue_store::primary::products::parse(&key.input, key.chain, Some(key.length), &step)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(index, product)| ProductWrite {
            publish: true,
            product,
            location: None,
            parents: Vec::new(),
            machine: Some("fixture-host".into()),
            output_index: index as u64,
        })
        .collect()
}

fn prefix(writes: &[ProductWrite]) -> BlobHash {
    writes[0]
        .product
        .reading()
        .unwrap()
        .delivery
        .unwrap()
        .prefix_hash
}

#[test]
fn export_callbacks_keep_exact_keys_and_prefixes_through_real_spool_restart_and_retry() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let key = key(12);
    let header = callback(&key, 0, ExportDelivery::seed(), false);
    let item = callback(&key, 1, prefix(&header), false);
    let complete = callback(&key, 2, prefix(&item), true);
    let first = sinex.prepare_export_products(&key, &header).unwrap();
    sinex.prepare_export_products(&key, &item).unwrap();
    sinex.prepare_export_products(&key, &complete).unwrap();
    assert_eq!(
        sinex
            .prepare_export_products(&key, &header)
            .unwrap()
            .identity,
        first.identity
    );
    drop(sinex);
    let reopened = client(root.path());
    let mut observed = Vec::new();
    reopened
        .visit_pending_export_readings(&key, &mut |reading| {
            reading.clone().into_step().validate_export()?;
            assert_eq!(reading.export_key.as_ref(), Some(&key));
            observed.push((
                reading.delivery.unwrap().ordinal,
                reading.parsed_upto,
                reading.progress.reached_end(),
            ));
            Ok(())
        })
        .unwrap();
    assert_eq!(observed, [(0, 0, false), (1, 0, false), (2, 12, true)]);
    assert_eq!(
        reopened
            .prepare_export_products(&key, &header)
            .unwrap()
            .identity,
        first.identity
    );
    let mut other = key.clone();
    other.length = 13;
    reopened
        .visit_pending_export_readings(&other, &mut |_| {
            panic!("other length borrowed pending callback")
        })
        .unwrap();
    other = key.clone();
    other.input.route = polylogue_model::Route::Copy;
    assert_eq!(
        reopened
            .prepare_export_products(&other, &header)
            .unwrap()
            .identity,
        first.identity
    );
    let mut count = 0;
    reopened
        .visit_pending_export_readings(&other, &mut |_| {
            count += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(count, 3);
}

#[test]
fn unbound_or_changed_export_products_are_refused_before_semantic_custody() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let key = key(12);
    let mut writes = callback(&key, 0, ExportDelivery::seed(), false);
    assert!(sinex.prepare_products(&writes).is_err());
    assert!(
        sinex
            .prepare_named_products("unbound-export", &writes)
            .is_err()
    );
    let mut other = key.clone();
    other.length = 13;
    assert!(sinex.prepare_export_products(&other, &writes).is_err());
    let Product::ParseProgress { counts, .. } = &mut writes[0].product else {
        panic!("control expected")
    };
    counts.records = 1;
    assert!(sinex.prepare_export_products(&key, &writes).is_err());
    assert_eq!(sinex.status().unwrap().prepared_materials, 0);
}

#[test]
fn export_source_length_and_callback_coordinate_distinguish_native_keys() {
    let key = key(12);
    let header = callback(&key, 0, ExportDelivery::seed(), false);
    let item = callback(&key, 1, prefix(&header), false);
    assert_ne!(
        crate::products::key(&header[0].product, 0).unwrap(),
        crate::products::key(&item.last().unwrap().product, 0).unwrap()
    );
    let other = callback(
        &ExportReadingKey {
            length: 13,
            ..key.clone()
        },
        0,
        ExportDelivery::seed(),
        false,
    );
    assert_eq!(prefix(&header), prefix(&other));
    assert_ne!(
        crate::products::key(&header[0].product, 0).unwrap(),
        crate::products::key(&other[0].product, 0).unwrap()
    );
    let payload = crate::products::payload(&header[0].product, 0).unwrap();
    let decoded = crate::products::decode(
        &crate::kinds::kind_reference("polylogue.parse_progress").unwrap(),
        payload,
    )
    .unwrap();
    assert_eq!(decoded, header[0].product);
    assert_eq!(
        crate::kinds::kind_reference("polylogue.parse_progress")
            .unwrap()
            .version,
        6
    );
    assert_eq!(crate::source_reference().version, 7);
    assert_eq!(
        crate::kinds::kind_reference("polylogue.query_run")
            .unwrap()
            .version,
        4
    );
    assert_eq!(
        crate::kinds::kind_reference("polylogue.query_run_outcome")
            .unwrap()
            .version,
        1
    );
}

#[test]
fn capture_sidecars_recover_original_callback_bodies() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let key = key(12);
    let mut captured_key = key.clone();
    captured_key.chain = 2;
    let captured_products = callback(&captured_key, 0, ExportDelivery::seed(), false);
    let reading = CapturedReading::from_product_group(
        &captured_key,
        captured_products.iter().map(|write| &write.product),
    )
    .unwrap();
    let mut captured = capture(b"source bytes", vec![reading.clone()]);
    captured.rows.revision.as_mut().unwrap().chain = 2;
    captured.rows.extents[0].chain = 2;
    sinex
        .prepare_input(&captured, body(root.path(), b"source bytes"))
        .unwrap();
    drop(sinex);
    let reopened = client(root.path());
    let mut sidecars = Vec::new();
    reopened
        .visit_pending_export_readings(&captured_key, &mut |value| {
            sidecars.push(value.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(sidecars, [reading]);
}

#[test]
fn unknown_export_sidecars_retain_provider_bytes_and_refuse_semantic_pending_progress() {
    use std::io::Read;
    for missing_key in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let key = key(12);
        let writes = callback(&key, 0, ExportDelivery::seed(), false);
        let mut reading =
            CapturedReading::from_product_group(&key, writes.iter().map(|write| &write.product))
                .unwrap();
        if missing_key {
            reading.export_key = None;
        } else {
            reading.counts.records = 1;
        }
        let sinex = client(root.path());
        let prepared = sinex
            .prepare_input(
                &capture(b"source bytes", vec![reading]),
                body(root.path(), b"source bytes"),
            )
            .unwrap();
        let polylogue_store::primary::PreparedInput::Pending { identity } = prepared else {
            panic!("offline custody must stay pending");
        };
        drop(sinex);
        let reopened = client(root.path());
        let mut bytes = Vec::new();
        reopened
            .queue()
            .unwrap()
            .body(&identity)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(bytes, b"source bytes");
        assert!(
            reopened
                .visit_pending_export_readings(&key, &mut |_| {
                    panic!("unknown export body became semantic progress")
                })
                .is_err()
        );
    }
}
