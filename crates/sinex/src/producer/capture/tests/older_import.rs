use super::*;
use polylogue_engine::{Engine, ImportOptions};
use polylogue_store::primary::{Confirmed, ConfirmedChange, InputProvenance, ReplayControl};
use polylogue_store::{Archive, older_archive};
use std::sync::Arc;

#[test]
fn ordinary_older_import_spools_original_history_and_reports_pending_classes() {
    let home = polylogue_testkit::ClaudeHome::new();
    let old_root = home.dir.path().join("older");
    let fixture = older_archive::fixture::standard(&old_root);
    let originals = older_archive::fixture::file_hashes(&old_root);
    let sdk = Arc::new(client(&home.dir.path().join("sinex")));
    let (archive, _lock) = Archive::open_primary(&home.root, &sdk.config.store).unwrap();
    let mut engine = Engine::on_sinex(
        home.config(1 << 20),
        archive.clone(),
        sdk.clone(),
        polylogue_testkit::stopped_attempt(),
    )
    .unwrap();
    let control = |control| Confirmed {
        position: 0,
        event: String::new(),
        output_index: 0,
        sinex_context: None,
        change: ConfirmedChange::Control(Box::new(control)),
    };
    engine
        .writer()
        .apply_confirmed(vec![control(ReplayControl::Reset {
            scope: "scope".into(),
        })])
        .unwrap();
    engine
        .writer()
        .apply_confirmed(vec![control(ReplayControl::Completion {
            position: 0,
            delivery: None,
        })])
        .unwrap();
    let options = ImportOptions {
        from_archive: Some(old_root.clone()),
        ..Default::default()
    };
    let report = engine
        .import_with_options(&[], &options)
        .unwrap()
        .older_archive
        .unwrap();
    assert!(report.pending);
    assert!(report.classes["input"].pending > 0);
    assert_eq!(report.classes["input"].stored, 0);
    assert!(report.classes["embedded"].refused > 0);
    assert!(report.classes["note"].refused > 0);
    let pending = sdk.pending_captures().unwrap();
    let restored: Vec<_> = pending
        .iter()
        .filter_map(|pending| match &pending.captured {
            Capture::Input(input) if input.provenance == InputProvenance::Restored => {
                Some((pending, input))
            }
            _ => None,
        })
        .collect();
    assert!(!restored.is_empty());
    let old =
        older_archive::OlderArchive::open(&old_root, &home.dir.path().join("scratch")).unwrap();
    for id in old.input_ids(None, 256).unwrap() {
        let expected = old.input(&id).unwrap();
        if expected.refusal.is_some() {
            continue;
        }
        let actual: Vec<_> = restored
            .iter()
            .filter(|(_, input)| input.rows.input == id)
            .map(|(_, input)| &input.rows)
            .collect();
        assert_eq!(
            actual
                .iter()
                .filter_map(|rows| rows.revision.clone())
                .collect::<Vec<_>>(),
            expected.revisions
        );
        assert_eq!(
            actual
                .iter()
                .flat_map(|rows| rows.extents.clone())
                .collect::<Vec<_>>(),
            expected.extents
        );
        assert_eq!(
            actual
                .iter()
                .filter_map(|rows| rows.observation.clone())
                .collect::<Vec<_>>(),
            expected.observations
        );
    }
    assert!(
        restored
            .iter()
            .any(|(_, input)| input.hash == fixture.file_blob)
    );
    assert!(pending.iter().any(
        |capture| matches!(&capture.captured,Capture::Blob{hash,..} if *hash==fixture.loose_blob)
    ));
    for (pending, input) in &restored {
        let original = old.input(&input.rows.input).unwrap();
        assert_eq!(input.rows.format, original.format);
        assert!(input.rows.acquisition.is_none());
        assert!(input.generation.is_none());
        assert!(input.basis.is_none());
        if let Some(revision) = &input.rows.revision {
            assert!(original.revisions.contains(revision));
        }
        for extent in &input.rows.extents {
            assert!(original.extents.contains(extent));
        }
        if let Some(observation) = &input.rows.observation {
            assert!(original.observations.contains(observation));
        }
        for body in sdk.capture_body(&pending.identity).unwrap() {
            assert_eq!(body.descriptor.hash.0, input.hash.0);
            assert_eq!(body.descriptor.decoded_len, input.length);
            let mut decoded = Vec::new();
            std::io::Read::read_to_end(
                &mut body
                    .descriptor
                    .open(body.file, None, hashed_blobs::ReadLimits::default())
                    .unwrap(),
                &mut decoded,
            )
            .unwrap();
            assert_eq!(BlobHash::of(&decoded), input.hash);
        }
    }
    let count = pending.len();
    let again = engine
        .import_with_options(&[], &options)
        .unwrap()
        .older_archive
        .unwrap();
    assert!(again.pending);
    assert_eq!(
        sdk.pending_captures().unwrap().len(),
        count,
        "retry keeps accepted capture identities"
    );
    // Supply confirmed material authority at the same boundary as Sinex. All
    // fixture bytes are encoded into one held segment, not invented hash lookups.
    let mut segment = hashed_blobs::SegmentWriter::open(
        &sdk.config.store,
        "segments/confirmed.hb",
        0,
        &[],
        0o600,
    )
    .unwrap();
    let mut bodies = Vec::new();
    for capture in &pending {
        let raw = matches!(&capture.captured,Capture::Input(input) if input.provenance==InputProvenance::Restored && input.raw_range.is_some())
            || matches!(&capture.captured, Capture::Blob { .. });
        if raw {
            for body in sdk.capture_body(&capture.identity).unwrap() {
                bodies.push(hashed_blobs::Prepared::from_own_encoding(
                    body.file,
                    body.descriptor,
                ));
            }
        }
    }
    let committed = segment
        .append_group(bodies, |_| Ok(None))
        .unwrap()
        .wait()
        .unwrap();
    segment.acknowledge(&committed).unwrap();
    let mut confirmed = Vec::new();
    for capture in &pending {
        let material = |hash: BlobHash, length: u64| polylogue_store::primary::Material {
            id: crate::material_id(&sdk.config.producer, &capture.identity).to_string(),
            hash,
            length,
            extents: vec![
                committed
                    .extents
                    .iter()
                    .find(|extent| extent.hash.0 == hash.0)
                    .expect("committed original")
                    .encode(),
            ],
        };
        let product = match &capture.captured {
            Capture::Input(input) if input.provenance == InputProvenance::Restored => {
                Some(Product::Input {
                    rows: Box::new(input.rows.clone()),
                    material: input.raw_range.map(|_| material(input.hash, input.length)),
                })
            }
            Capture::Blob { hash, length } => Some(Product::Blob(material(*hash, *length))),
            _ => None,
        };
        if let Some(product) = product {
            confirmed.push(Confirmed {
                position: 1,
                event: format!("confirmed:{}", capture.identity),
                output_index: 0,
                sinex_context: None,
                change: ConfirmedChange::Live(Box::new(product)),
            });
        }
    }
    confirmed.push(control(ReplayControl::Completion {
        position: 1,
        delivery: None,
    }));
    engine.writer().apply_confirmed(confirmed).unwrap();
    let applied = engine
        .import_with_options(&[], &options)
        .unwrap()
        .older_archive
        .unwrap();
    assert_eq!(applied.classes["input"].pending, 0);
    assert_eq!(applied.classes["input"].stored, 0);
    assert_eq!(
        applied.classes["input"].present,
        report.classes["input"].pending
    );
    assert_eq!(
        sdk.pending_captures().unwrap().len(),
        count,
        "confirmed retry does not create generic copies of captured originals"
    );
    assert_eq!(older_archive::fixture::file_hashes(&old_root), originals);
    assert!(!home.root.join("archive.db").exists());
}
