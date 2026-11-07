use super::*;

fn body() -> BlobHash {
    BlobHash::of(b"actual semantic body")
}

#[test]
fn export_prefix_codec_binds_the_frozen_domain_and_json_tuple() {
    let seed = ExportDelivery::seed();
    assert_eq!(seed, BlobHash::of(b"polylogue.export-delivery.prefix.v1"));
    let delivery =
        ExportDelivery::new(0, ExportDeliverySource::ContainerHeader, body(), seed).unwrap();
    let mut expected = b"polylogue.export-delivery.link.v1\0".to_vec();
    expected.extend_from_slice(
        format!(
            "[1,0,{{\"kind\":\"container_header\"}},\"{}\",\"{}\"]",
            body(),
            seed
        )
        .as_bytes(),
    );
    assert_eq!(delivery.prefix_hash, BlobHash::of(&expected));
    let mut changed = delivery;
    changed.body_hash = BlobHash::of(b"different body");
    assert_eq!(changed.validate(), Err(ExportDeliveryError::InvalidPrefix));
}

#[test]
fn export_ordinals_follow_callbacks_while_source_byte_ranges_can_move_backwards() {
    let header = ExportDelivery::new(
        0,
        ExportDeliverySource::ContainerHeader,
        body(),
        ExportDelivery::seed(),
    )
    .unwrap();
    let first = ExportDelivery::new(
        1,
        ExportDeliverySource::Item {
            start: 100,
            end: 120,
        },
        body(),
        header.prefix_hash,
    )
    .unwrap();
    let second = ExportDelivery::new(
        2,
        ExportDeliverySource::Item { start: 10, end: 20 },
        body(),
        first.prefix_hash,
    )
    .unwrap();
    let complete = ExportDelivery::new(
        3,
        ExportDeliverySource::Complete {
            length: 200,
            item_count: 2,
            delivery_count: 3,
        },
        body(),
        second.prefix_hash,
    )
    .unwrap();
    let cursor = ExportCursor::from_first(&header)
        .unwrap()
        .advance(&first)
        .unwrap()
        .advance(&second)
        .unwrap()
        .advance(&complete)
        .unwrap();
    assert_eq!(cursor.last_ordinal, 3);
    assert_eq!(cursor.prefix_hash, complete.prefix_hash);
    assert_eq!(
        serde_json::to_value(cursor)
            .unwrap()
            .as_object()
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn verified_empty_exports_and_header_only_exports_have_distinct_valid_completion_counts() {
    let empty = ExportDelivery::new(
        0,
        ExportDeliverySource::Complete {
            length: 0,
            item_count: 0,
            delivery_count: 0,
        },
        body(),
        ExportDelivery::seed(),
    )
    .unwrap();
    assert_eq!(ExportCursor::from_first(&empty).unwrap().last_ordinal, 0);
    let header = ExportDelivery::new(
        0,
        ExportDeliverySource::ContainerHeader,
        body(),
        ExportDelivery::seed(),
    )
    .unwrap();
    let end = ExportDelivery::new(
        1,
        ExportDeliverySource::Complete {
            length: 10,
            item_count: 0,
            delivery_count: 1,
        },
        body(),
        header.prefix_hash,
    )
    .unwrap();
    header.cursor().unwrap().advance(&end).unwrap();
}

#[test]
fn export_shape_refuses_reversed_ranges_inconsistent_counts_and_final_ordinal() {
    for source in [
        ExportDeliverySource::Complete {
            length: 10,
            item_count: 2,
            delivery_count: 1,
        },
        ExportDeliverySource::Complete {
            length: 10,
            item_count: 0,
            delivery_count: 2,
        },
    ] {
        assert_eq!(
            source.validate(),
            Err(ExportDeliveryError::InvalidCompletionCounts)
        );
    }
    assert_eq!(
        ReadingFootprint::Bytes { start: 2, end: 1 }.validate(),
        Err(ExportDeliveryError::InvalidSourceRange)
    );
    assert_eq!(
        ExportDeliverySource::Item { start: 2, end: 1 }.validate(),
        Err(ExportDeliveryError::InvalidSourceRange)
    );
    assert_eq!(
        ExportDelivery::new(
            2,
            ExportDeliverySource::Complete {
                length: 10,
                item_count: 1,
                delivery_count: 1
            },
            body(),
            ExportDelivery::seed()
        ),
        Err(ExportDeliveryError::InvalidOrdinal)
    );
}

#[test]
fn export_cursor_extensions_refuse_replay_gaps_forks_and_overflow() {
    let first = ExportDelivery::new(
        0,
        ExportDeliverySource::Item { start: 0, end: 1 },
        body(),
        ExportDelivery::seed(),
    )
    .unwrap();
    let cursor = ExportCursor::from_first(&first).unwrap();
    assert_eq!(
        cursor.advance(&first),
        Err(ExportDeliveryError::InvalidOrdinal)
    );
    let gap = ExportDelivery::new(
        2,
        ExportDeliverySource::Item { start: 2, end: 3 },
        body(),
        cursor.prefix_hash,
    )
    .unwrap();
    assert_eq!(
        cursor.advance(&gap),
        Err(ExportDeliveryError::InvalidOrdinal)
    );
    let fork = ExportDelivery::new(
        1,
        ExportDeliverySource::Item { start: 1, end: 2 },
        body(),
        BlobHash::of(b"wrong prefix"),
    )
    .unwrap();
    assert_eq!(
        cursor.advance(&fork),
        Err(ExportDeliveryError::InvalidPrefix)
    );
    let last = ExportCursor {
        last_ordinal: u64::MAX,
        ..cursor
    };
    assert_eq!(
        last.next_ordinal(),
        Err(ExportDeliveryError::OrdinalOverflow)
    );
}

#[test]
fn export_control_round_trips_without_becoming_a_native_parser_checkpoint() {
    let delivery = ExportDelivery::new(
        0,
        ExportDeliverySource::ContainerHeader,
        body(),
        ExportDelivery::seed(),
    )
    .unwrap();
    let value = serde_json::to_value(delivery).unwrap();
    assert_eq!(
        serde_json::from_value::<ExportDelivery>(value.clone()).unwrap(),
        delivery
    );
    assert!(serde_json::from_value::<super::super::ParserCheckpoint>(value.clone()).is_err());
    let mut unknown = value;
    unknown["native_admission"] = serde_json::json!(true);
    assert!(serde_json::from_value::<ExportDelivery>(unknown).is_err());
    let mut unsupported = delivery;
    unsupported.version = 2;
    assert_eq!(
        unsupported.validate(),
        Err(ExportDeliveryError::UnsupportedVersion { version: 2 })
    );
    let mut cursor = delivery.cursor().unwrap();
    cursor.version = 2;
    assert_eq!(
        cursor.validate(),
        Err(ExportDeliveryError::UnsupportedVersion { version: 2 })
    );
    serde_json::to_value(schemars::schema_for!(ExportDelivery)).unwrap();
    serde_json::to_value(schemars::schema_for!(ReadingFootprint)).unwrap();
}
