use super::*;
use polylogue_model::batch::{ExportDelivery, ExportDeliverySource};
use polylogue_model::{Output, RecordCounts};
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
fn unbound_export_products_are_refused_before_semantic_custody() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let key = key(12);
    let writes = callback(&key, 0, ExportDelivery::seed(), false);
    assert!(sinex.prepare_products(&writes).is_err());
    assert!(
        sinex
            .prepare_named_products("unbound-export", &writes)
            .is_err()
    );
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
    assert_eq!(crate::source_reference().version, crate::SOURCE_VERSION);
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
