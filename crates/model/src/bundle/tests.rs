use super::*;

fn manifest() -> Manifest {
    Manifest {
        format: FORMAT.into(),
        version: VERSION,
        query: format!("query:{}", "a".repeat(64)),
        query_text: "sessions".into(),
        ran_at: Timestamp::from_unix_ms(1).unwrap(),
        read_at: ChangePosition {
            generation: "generation".into(),
            index: 2,
            archive: 3,
        },
        caught_up: false,
        counts: Counts::default(),
        flags: Flags {
            format: Format::Json,
            view: crate::read::View::Full,
            with_blobs: false,
            inherited: true,
            assertions: true,
        },
        enabled_variants: vec![],
        author: Author::Operator,
        inventory: None,
    }
}

#[test]
fn bundle_header_refuses_earlier_and_newer_formats_before_contents() {
    let mut value = manifest();
    assert!(value.validate().is_ok());
    for version in [0, 2, u32::MAX] {
        value.version = version;
        assert_eq!(
            value.validate(),
            Err(BundleError::UnsupportedVersion { found: version })
        );
    }
    value.format = "polylogue-export".into();
    assert_eq!(
        value.validate(),
        Err(BundleError::WrongFormat("polylogue-export".into()))
    );
}

#[test]
fn inventory_names_are_relative_and_cannot_escape_or_alias_members() {
    for path in [
        "",
        "/absolute",
        "../outside",
        "a/../b",
        "./member",
        "a//b",
        "a/",
        "a\\b",
        "a\0b",
    ] {
        let file = FileDescriptor {
            path: path.into(),
            hash: BlobHash::of(b"member"),
            size: 6,
        };
        assert_eq!(file.validate(), Err(BundleError::InvalidPath), "{path:?}");
    }
    let file = FileDescriptor {
        path: "codex/native%2Fid.json".into(),
        hash: BlobHash::of(b"member"),
        size: 6,
    };
    assert!(file.validate().is_ok());
}

#[test]
fn event_identity_keeps_every_claim_and_validates_exact_reference_and_time() {
    let claim = |input: &str| EventSourceClaim {
        input: input.into(),
        source: "codex".into(),
        source_chain: 1,
        source_start: Some(10),
        source_end: Some(20),
    };
    let mut identity = ImportedEventIdentity {
        id: "event-id".into(),
        sources: vec![claim("b"), claim("a"), claim("b")],
    };
    identity.canonicalize();
    assert_eq!(identity.sources, vec![claim("a"), claim("b")]);
    let event: Event = serde_json::from_value(
        serde_json::json!({ "kind": "quota_report", "data": {"remaining": 7} }),
    )
    .unwrap();
    let mut record = InputEventRecord {
        reference: "event:event-id".into(),
        commit: 4,
        identity,
        event,
    };
    assert!(record.validate().is_ok());
    record.event.at = Timestamp::from_unix_ms(1);
    assert_eq!(
        record.validate(),
        Err(BundleError::Time(crate::TimeEvidenceError::UnknownQuality))
    );
    record
        .event
        .set_time(
            record.event.at,
            crate::TimeEvidence::intrinsic(crate::Precision::Instant),
        )
        .unwrap();
    assert!(record.validate().is_ok());
    record.reference = "event:another".into();
    assert_eq!(record.validate(), Err(BundleError::EventReferenceMismatch));
}
